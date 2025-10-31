import pytest
import asyncio
import sys
# sys.path.append('../agents')  # Removed; use package imports instead
from agents.main import PatientData, EligibilityAgent, PrescriptionAgent, FederatedLearningCoordinator, DiagnosisModel, Fernet, hashlib

@pytest.fixture
def solana_endpoint():
    return "https://api.devnet.solana.com"

@pytest.fixture
def sample_patients():
    patients = []
    for i in range(105):  # 100+ sims
        patient = PatientData(
            patient_id=f"PATIENT_{i:03d}",
            encrypted_data=Fernet.generate_key(),
            ipfs_cid=f"Qm{hashlib.sha256(f'patient{i}'.encode()).hexdigest()[:44]}",
            data_hash=hashlib.sha256(f"medical_history_{i}".encode()).hexdigest()
        )
        patients.append(patient)
    return patients

@pytest.mark.asyncio
async def test_eligibility_check(sample_patients, solana_endpoint):
    agent = EligibilityAgent(solana_endpoint)
    for patient in sample_patients[:100]:
        result = await agent.check_insurance_coverage(patient, "PROC001")
        assert result["eligible"] is True
        assert result["privacy_preserved"] is True
    assert len(sample_patients) > 100

@pytest.mark.asyncio
async def test_prescription_validation(sample_patients, solana_endpoint):
    agent = PrescriptionAgent(solana_endpoint)
    for patient in sample_patients[:100]:
        result = await agent.validate_prescription(patient, "DRUG001")
        assert result["valid"] is True
        assert result["zk_proof_verified"] is True

@pytest.mark.asyncio
async def test_federated_learning(sample_patients):
    model = DiagnosisModel()
    coordinator = FederatedLearningCoordinator(model, num_agents=3)
    result = await coordinator.train_round(sample_patients[:105])
    assert result["participants"] == 3
    assert result["round"] == 1

if __name__ == "__main__":
    pytest.main(["-v"])
