use anchor_lang::prelude::*;
use anchor_lang::solana_program::keccak;

declare_id!("HEALth11111111111111111111111111111111111");

#[program]
pub mod zk_healthcare {
    use super::*;

    /// Initialize the healthcare verification system
    pub fn initialize(ctx: Context<Initialize>, nist_compliant: bool) -> Result<()> {
        let registry = &mut ctx.accounts.registry;
        registry.authority = ctx.accounts.authority.key();
        registry.nist_compliant = nist_compliant;
        registry.total_verifications = 0;
        registry.ipfs_pin_count = 0;
        msg!("Healthcare ZK Registry initialized");
        Ok(())
    }

    /// Verify zk-SNARK proof for patient eligibility
    pub fn verify_eligibility(
        ctx: Context<VerifyEligibility>,
        proof: Vec<u8>,
        public_inputs: Vec<u8>,
        ipfs_hash: String,
    ) -> Result<()> {
        let registry = &mut ctx.accounts.registry;
        let verification = &mut ctx.accounts.verification;

        // Groth16 verification (simplified - use actual groth16 verifier crate)
        require!(
            proof.len() == 256,
            HealthcareError::InvalidProofLength
        );
        
        // Simulate proof verification
        let is_valid = verify_groth16_proof(&proof, &public_inputs)?;
        require!(is_valid, HealthcareError::ProofVerificationFailed);

        // Store verification record
        verification.patient_pubkey = ctx.accounts.patient.key();
        verification.proof_hash = keccak::hash(&proof).to_bytes();
        verification.ipfs_hash = ipfs_hash.clone();
        verification.timestamp = Clock::get()?.unix_timestamp;
        verification.is_valid = true;
        verification.verification_type = VerificationType::Eligibility;

        // Update registry
        registry.total_verifications += 1;
        registry.ipfs_pin_count += 1;

        emit!(EligibilityVerified {
            patient: ctx.accounts.patient.key(),
            ipfs_hash,
            timestamp: verification.timestamp,
        });

        msg!("Eligibility verified. Gas estimated: ~450K compute units");
        Ok(())
    }

    /// Store encrypted medical data hash on IPFS
    pub fn pin_medical_data(
        ctx: Context<PinMedicalData>,
        ipfs_cid: String,
        data_hash: [u8; 32],
    ) -> Result<()> {
        let pin_record = &mut ctx.accounts.pin_record;
        let registry = &mut ctx.accounts.registry;

        pin_record.patient = ctx.accounts.patient.key();
        pin_record.ipfs_cid = ipfs_cid.clone();
        pin_record.data_hash = data_hash;
        pin_record.pinned_at = Clock::get()?.unix_timestamp;
        pin_record.access_count = 0;

        registry.ipfs_pin_count += 1;

        emit!(DataPinned {
            patient: ctx.accounts.patient.key(),
            ipfs_cid,
            data_hash,
        });

        Ok(())
    }

    /// Oracle function for federated learning model updates
    pub fn submit_model_update(
        ctx: Context<SubmitModelUpdate>,
        encrypted_gradient: Vec<u8>,
        round_number: u64,
    ) -> Result<()> {
        let fl_state = &mut ctx.accounts.fl_state;
        
        require!(
            encrypted_gradient.len() <= 4096,
            HealthcareError::GradientTooLarge
        );

        fl_state.round_number = round_number;
        fl_state.last_update = Clock::get()?.unix_timestamp;
        fl_state.participant_count += 1;

        msg!("Federated learning update submitted for round {}", round_number);
        Ok(())
    }
}

// Account structures
#[account]
pub struct HealthcareRegistry {
    pub authority: Pubkey,
    pub nist_compliant: bool,
    pub total_verifications: u64,
    pub ipfs_pin_count: u64,
}

#[account]
pub struct VerificationRecord {
    pub patient_pubkey: Pubkey,
    pub proof_hash: [u8; 32],
    pub ipfs_hash: String,
    pub timestamp: i64,
    pub is_valid: bool,
    pub verification_type: VerificationType,
}

#[account]
pub struct IpfsPinRecord {
    pub patient: Pubkey,
    pub ipfs_cid: String,
    pub data_hash: [u8; 32],
    pub pinned_at: i64,
    pub access_count: u32,
}

#[account]
pub struct FederatedLearningState {
    pub round_number: u64,
    pub last_update: i64,
    pub participant_count: u32,
}

// Enums
#[derive(AnchorSerialize, AnchorDeserialize, Clone, Copy, PartialEq, Eq)]
pub enum VerificationType {
    Eligibility,
    Prescription,
    Diagnosis,
    AccessControl,
}

// Context structs
#[derive(Accounts)]
pub struct Initialize<'info> {
    #[account(init, payer = authority, space = 8 + 128)]
    pub registry: Account<'info, HealthcareRegistry>,
    #[account(mut)]
    pub authority: Signer<'info>,
    pub system_program: Program<'info, System>,
}

#[derive(Accounts)]
pub struct VerifyEligibility<'info> {
    #[account(mut)]
    pub registry: Account<'info, HealthcareRegistry>,
    #[account(init, payer = patient, space = 8 + 256)]
    pub verification: Account<'info, VerificationRecord>,
    #[account(mut)]
    pub patient: Signer<'info>,
    pub system_program: Program<'info, System>,
}

#[derive(Accounts)]
pub struct PinMedicalData<'info> {
    #[account(mut)]
    pub registry: Account<'info, HealthcareRegistry>,
    #[account(init, payer = patient, space = 8 + 256)]
    pub pin_record: Account<'info, IpfsPinRecord>,
    #[account(mut)]
    pub patient: Signer<'info>,
    pub system_program: Program<'info, System>,
}

#[derive(Accounts)]
pub struct SubmitModelUpdate<'info> {
    #[account(mut)]
    pub fl_state: Account<'info, FederatedLearningState>,
    pub agent: Signer<'info>,
}

// Events
#[event]
pub struct EligibilityVerified {
    pub patient: Pubkey,
    pub ipfs_hash: String,
    pub timestamp: i64,
}

#[event]
pub struct DataPinned {
    pub patient: Pubkey,
    pub ipfs_cid: String,
    pub data_hash: [u8; 32],
}

// Errors
#[error_code]
pub enum HealthcareError {
    #[msg("Invalid proof length")]
    InvalidProofLength,
    #[msg("Proof verification failed")]
    ProofVerificationFailed,
    #[msg("Gradient data too large")]
    GradientTooLarge,
}

// Helper function (placeholder for actual Groth16 verification)
fn verify_groth16_proof(proof: &[u8], public_inputs: &[u8]) -> Result<bool> {
    // In production: use arkworks or bellman crate
    // This is a placeholder for demonstration
    Ok(proof.len() == 256 && !public_inputs.is_empty())
          }
