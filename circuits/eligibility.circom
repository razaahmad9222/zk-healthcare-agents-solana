pragma circom 2.0.0.0;

// Patient eligibility verification circuit
template PatientEligibility() {
    // Private inputs (patient data)
    signal input patientID;
    signal input dateOfBirth;
    signal input insurancePolicyNum;
    signal input medicalHistoryHash;
    
    // Public inputs (verification criteria)
    signal input requiredAge;
    signal input insuranceProviderID;
    signal input currentTimestamp;
    
    // Output: 1 if eligible, 0 otherwise
    signal output isEligible;
    
    // Intermediate signals
    signal ageInYears;
    signal ageCheck;
    signal insuranceCheck;
    signal policyValidHash;
    
    // Age calculation (simplified)
    component ageCalc = CalculateAge();
    ageCalc.dob <== dateOfBirth;
    ageCalc.currentTime <== currentTimestamp;
    ageInYears <== ageCalc.age;
    
    // Age verification
    component ageComparator = GreaterEqThan(32);
    ageComparator.in[0] <== ageInYears;
    ageComparator.in[1] <== requiredAge;
    ageCheck <== ageComparator.out;
    
    // Insurance verification (hash-based)
    component policyHasher = Poseidon(2);
    policyHasher.inputs[0] <== insurancePolicyNum;
    policyHasher.inputs[1] <== insuranceProviderID;
    policyValidHash <== policyHasher.out;
    
    component insuranceValidator = IsEqual();
    insuranceValidator.in[0] <== policyValidHash;
    insuranceValidator.in[1] <== medicalHistoryHash; // Linked validation
    insuranceCheck <== insuranceValidator.out;
    
    // Final eligibility (AND gate)
    isEligible <== ageCheck * insuranceCheck;
}

template CalculateAge() {
    signal input dob;
    signal input currentTime;
    signal output age;
    
    signal timeDiff;
    timeDiff <== currentTime - dob;
    
    // Convert seconds to years (365.25 days)
    component yearConverter = Divider(64);
    yearConverter.dividend <== timeDiff;
    yearConverter.divisor <== 31557600; // Seconds per year
    age <== yearConverter.quotient;
}

component main = PatientEligibility();
