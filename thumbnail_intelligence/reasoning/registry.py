"""
registry.py
===========

Pluggable Reasoner Registry for the Strategic Reasoning Coordinator.
Enables dynamic registration, lookup, contract validation, dependency verification,
and deterministic topological execution ordering without hardcoding reasoners.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Set

from thumbnail_intelligence.reasoning.exceptions import (
    CircularDependencyError,
    DuplicateReasonerError,
    InvalidReasonerError,
    MissingDependencyError,
    ReasonerNotFoundError,
)
from thumbnail_intelligence.reasoning.interfaces import BaseReasoner

_SEMVER_REGEX = re.compile(
    r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)(?:-((?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*)(?:\.(?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*))*))?(?:\+([0-9a-zA-Z-]+(?:\.[0-9a-zA-Z-]+)*))?$"
)


class ReasonerRegistry:
    """
    Central registry for strategic reasoning modules.
    Provides dependency checking and topological execution ordering.
    """

    def __init__(self) -> None:
        self._reasoners: Dict[str, BaseReasoner] = {}

    def register(self, reasoner: BaseReasoner, override: bool = False) -> None:
        """
        Register a strategic reasoner.
        Validates contract and interface compliance.

        Args:
            reasoner: An instance implementing BaseReasoner.
            override: If True, allows replacing an existing reasoner with the same name.

        Raises:
            InvalidReasonerError: If reasoner fails interface or contract validation.
            DuplicateReasonerError: If reasoner name already exists and override is False.
        """
        self.validate_reasoner(reasoner)

        name = reasoner.name
        if name in self._reasoners and not override:
            existing = self._reasoners[name]
            raise DuplicateReasonerError(
                reasoner_name=name,
                existing_version=existing.version,
            )

        self._reasoners[name] = reasoner

    def unregister(self, name: str) -> bool:
        """
        Unregister a reasoner by name.

        Returns:
            True if reasoner was removed, False if not found.
        """
        if name in self._reasoners:
            del self._reasoners[name]
            return True
        return False

    def get(self, name: str) -> Optional[BaseReasoner]:
        """Retrieve a reasoner by name, or None if not registered."""
        return self._reasoners.get(name)

    def get_required(self, name: str) -> BaseReasoner:
        """
        Retrieve a reasoner by name, raising ReasonerNotFoundError if missing.

        Raises:
            ReasonerNotFoundError: If reasoner is not registered.
        """
        if name not in self._reasoners:
            raise ReasonerNotFoundError(
                reasoner_name=name,
                available_reasoners=self.list_names(),
            )
        return self._reasoners[name]

    def list(self) -> List[BaseReasoner]:
        """Return a list of all registered reasoners."""
        return list(self._reasoners.values())

    def list_names(self) -> List[str]:
        """Return a list of all registered reasoner names."""
        return list(self._reasoners.keys())

    def has(self, name: str) -> bool:
        """Check if a reasoner with the given name is registered."""
        return name in self._reasoners

    def count(self) -> int:
        """Return the number of registered reasoners."""
        return len(self._reasoners)

    def clear(self) -> None:
        """Remove all registered reasoners."""
        self._reasoners.clear()

    def validate_reasoner(self, reasoner: BaseReasoner) -> None:
        """
        Validate reasoner contract, interface implementation, name, version, and dependencies.

        Raises:
            InvalidReasonerError: If any contract invariant fails.
        """
        errors: List[str] = []

        if not isinstance(reasoner, BaseReasoner):
            errors.append("Object does not inherit from BaseReasoner")
            raise InvalidReasonerError(
                reasoner_name=getattr(reasoner, "name", "unknown"),
                validation_errors=errors,
            )

        try:
            contract = reasoner.contract
            if contract is None:
                errors.append("contract property returned None")
        except Exception as exc:
            errors.append(f"Failed to access contract property: {exc}")
            raise InvalidReasonerError(
                reasoner_name=getattr(reasoner, "name", "unknown"),
                validation_errors=errors,
            )

        # Name validation
        name = getattr(reasoner, "name", None)
        if not name or not isinstance(name, str) or not name.strip():
            errors.append("Reasoner name must be a non-empty string")
        else:
            name = name.strip()

        # Version validation
        version = getattr(reasoner, "version", "")
        if not version or not isinstance(version, str) or not _SEMVER_REGEX.match(version):
            errors.append(f"Reasoner version '{version}' is not a valid SemVer string")

        # Dependencies validation
        deps = getattr(reasoner, "dependencies", None)
        if not isinstance(deps, list):
            errors.append("Reasoner dependencies must be a list of strings")
        else:
            for dep in deps:
                if not isinstance(dep, str) or not dep.strip():
                    errors.append("Dependency names must be non-empty strings")
                elif dep.strip() == name:
                    errors.append(f"Reasoner cannot depend on itself ('{name}')")

        if errors:
            raise InvalidReasonerError(
                reasoner_name=name or "invalid",
                validation_errors=errors,
            )

    def check_dependencies(self) -> None:
        """
        Verify that all dependencies declared by registered reasoners exist in the registry.

        Raises:
            MissingDependencyError: If a required upstream reasoner is not registered.
        """
        available = self.list_names()
        for name, reasoner in self._reasoners.items():
            for dep in reasoner.dependencies:
                dep_name = dep.strip()
                if dep_name not in self._reasoners:
                    raise MissingDependencyError(
                        reasoner_name=name,
                        missing_dependency=dep_name,
                        available_reasoners=available,
                    )

    def get_execution_order(self) -> List[BaseReasoner]:
        """
        Compute a deterministic topological execution order of registered reasoners.
        Ensures all dependencies run before their dependent reasoners.

        Returns:
            List of BaseReasoner instances sorted in dependency order.

        Raises:
            MissingDependencyError: If an upstream dependency is missing.
            CircularDependencyError: If circular dependencies are detected.
        """
        self.check_dependencies()

        if not self._reasoners:
            return []

        # Build in-degree counts and adjacency graph
        # Edge u -> v means reasoner u is a dependency of v (u must execute before v)
        adj: Dict[str, Set[str]] = {name: set() for name in self._reasoners}
        in_degree: Dict[str, int] = {name: 0 for name in self._reasoners}

        for name, reasoner in self._reasoners.items():
            for dep in reasoner.dependencies:
                dep_name = dep.strip()
                adj[dep_name].add(name)
                in_degree[name] += 1

        # Kahn's algorithm with deterministic tie-breaking (sorted by name)
        queue: List[str] = sorted([name for name, deg in in_degree.items() if deg == 0])
        ordered_names: List[str] = []

        while queue:
            curr = queue.pop(0)
            ordered_names.append(curr)

            # Decrement in-degree for all downstream dependents
            for dependent in sorted(adj[curr]):
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    queue.append(dependent)
                    queue.sort()

        if len(ordered_names) < len(self._reasoners):
            # Find cycle path for diagnostic error message
            cycle_nodes = [name for name, deg in in_degree.items() if deg > 0]
            cycle_path = self._find_cycle_path(cycle_nodes, adj)
            raise CircularDependencyError(cycle_path=cycle_path)

        return [self._reasoners[name] for name in ordered_names]

    def _find_cycle_path(self, cycle_nodes: List[str], adj: Dict[str, Set[str]]) -> List[str]:
        """Find a representative cycle path among the cycle nodes."""
        visited: Set[str] = set()
        stack: List[str] = []

        def dfs(node: str) -> Optional[List[str]]:
            if node in stack:
                idx = stack.index(node)
                return stack[idx:] + [node]
            if node in visited:
                return None

            visited.add(node)
            stack.append(node)
            for neighbor in sorted(adj.get(node, [])):
                if neighbor in cycle_nodes:
                    res = dfs(neighbor)
                    if res is not None:
                        return res
            stack.pop()
            return None

        for start in sorted(cycle_nodes):
            path = dfs(start)
            if path is not None:
                return path

        return cycle_nodes

    def clone(self) -> ReasonerRegistry:
        """Create an independent copy of this registry."""
        new_registry = ReasonerRegistry()
        for reasoner in self._reasoners.values():
            new_registry.register(reasoner)
        return new_registry
