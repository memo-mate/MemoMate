"""Simplified plugin loader using importlib."""

import importlib
import inspect
import sys
from pathlib import Path
from typing import Any

from app.core import logger

from .base import RerankPlugin


class PluginLoader:
    """Simplified plugin loader using importlib."""

    def __init__(self, plugin_package: str = "app.plugins.rerank.plugins"):
        """Initialize plugin loader.

        Args:
            plugin_package: Python package containing plugins
        """
        self.plugin_package = plugin_package
        self._plugin_classes: dict[str, type[RerankPlugin]] = {}
        self._loaded_modules: dict[str, Any] = {}

    def discover_plugins(self, reload: bool = False) -> None:
        """Discover plugins from the package using importlib.

        Args:
            reload: If True, reload all modules
        """
        if reload:
            self._plugin_classes.clear()
            self._reload_all_modules()

        try:
            # Import the plugins package
            package = importlib.import_module(self.plugin_package)

            # Get package path - handle both regular and namespace packages
            if hasattr(package, "__file__") and package.__file__ is not None:
                package_path = Path(package.__file__).parent
            elif hasattr(package, "__path__") and package.__path__:
                # Handle namespace packages
                package_path = Path(package.__path__[0])
            else:
                logger.error(f"Cannot determine package path for {self.plugin_package}")
                return

            # Dynamically scan for all Python files in the directory
            python_files = list(package_path.glob("*.py"))
            for py_file in python_files:
                if py_file.name.startswith("_") or py_file.name == "__init__.py":
                    continue

                module_name = f"{self.plugin_package}.{py_file.stem}"
                self._load_module(module_name, reload)

            logger.info(
                f"Discovered {len(self._plugin_classes)} plugins from {len(python_files)} modules in {package_path}"
            )

        except Exception as e:
            logger.error(f"Failed to discover plugins: {e}")

    def _load_module(self, module_name: str, reload: bool = False) -> None:
        """Load or reload a module and extract plugins.

        Args:
            module_name: Full module name
            reload: Whether to reload the module
        """
        try:
            if reload and module_name in sys.modules:
                # Reload existing module
                module = importlib.reload(sys.modules[module_name])
            else:
                # Import new module
                module = importlib.import_module(module_name)

            self._loaded_modules[module_name] = module

            # Extract plugin classes
            for name, obj in inspect.getmembers(module, inspect.isclass):
                if self._is_plugin_class(obj):
                    self._plugin_classes[name] = obj
                    logger.debug(f"Registered plugin: {name}")

        except Exception as e:
            logger.error(f"Failed to load module {module_name}: {e}")

    def _reload_all_modules(self) -> None:
        """Reload all previously loaded modules."""
        for module_name in list(self._loaded_modules.keys()):
            if module_name in sys.modules:
                try:
                    importlib.reload(sys.modules[module_name])
                    logger.debug(f"Reloaded module: {module_name}")
                except Exception as e:
                    logger.error(f"Failed to reload module {module_name}: {e}")

    def _is_plugin_class(self, cls: type) -> bool:
        """Check if a class is a valid plugin class."""
        return (
            inspect.isclass(cls)
            and issubclass(cls, RerankPlugin)
            and cls != RerankPlugin
            and not inspect.isabstract(cls)
        )

    def create_plugin(
        self,
        plugin_name: str,
        config: dict[str, Any] | None = None,
    ) -> RerankPlugin:
        """Create a plugin instance."""
        if plugin_name not in self._plugin_classes:
            # Try to discover plugins if not found
            self.discover_plugins()

        if plugin_name not in self._plugin_classes:
            raise ValueError(f"Plugin not found: {plugin_name}")

        plugin_class = self._plugin_classes[plugin_name]
        return plugin_class(config=config)

    def reload_plugins(self) -> None:
        """Reload all plugins."""
        logger.info("Reloading all plugins...")
        self.discover_plugins(reload=True)

    def list_plugins(self) -> list[str]:
        """Get list of available plugin names."""
        return list(self._plugin_classes.keys())


# Global plugin loader instance
_plugin_loader: PluginLoader | None = None


def get_plugin_loader() -> PluginLoader:
    """Get the global plugin loader instance."""
    global _plugin_loader
    if _plugin_loader is None:
        _plugin_loader = PluginLoader()
        _plugin_loader.discover_plugins()
    return _plugin_loader
