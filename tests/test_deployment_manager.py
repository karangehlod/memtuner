
import pytest
from benchmark.memory.production.deployment_manager import DeploymentManager

@pytest.fixture
def manager():
    return DeploymentManager()

class TestDeploymentManager:
    def test_init(self, manager):
        assert manager.environment == "production"
    
    def test_prepare_deployment(self, manager):
        pkg = manager.prepare_deployment()
        assert pkg.version is not None
    
    def test_validate_deployment(self, manager):
        result = manager.validate_deployment()
        assert result["valid"]
    
    def test_execute_deployment(self, manager):
        result = manager.execute_deployment()
        assert result["deployed"]
    
    def test_rollback(self, manager):
        result = manager.rollback()
        assert result["rolled_back"]
    
    def test_package_contains_files(self, manager):
        pkg = manager.prepare_deployment()
        assert len(pkg.files) > 0
    
    def test_package_checksums(self, manager):
        pkg = manager.prepare_deployment()
        assert len(pkg.checksums) > 0
    
    def test_validate_checks_passed(self, manager):
        result = manager.validate_deployment()
        assert result["checks_passed"] > 0
    
    def test_deployment_tracked(self, manager):
        manager.execute_deployment()
        assert len(manager._deployments) == 1
    
    def test_version_in_package(self, manager):
        pkg = manager.prepare_deployment()
        assert pkg.version == "1.0.0"
    
    def test_env_specified(self):
        m = DeploymentManager(environment="staging")
        assert m.environment == "staging"
    
    def test_rollback_returns_version(self, manager):
        result = manager.rollback()
        assert "previous_version" in result
    
    def test_deployment_success(self, manager):
        manager.execute_deployment()
        manager.execute_deployment()
        assert len(manager._deployments) == 2
    
    def test_package_version_consistent(self, manager):
        pkg1 = manager.prepare_deployment()
        pkg2 = manager.prepare_deployment()
        assert pkg1.version == pkg2.version
