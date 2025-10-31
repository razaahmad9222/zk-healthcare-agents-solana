#!/bin/bash

set -e

CIRCUIT="eligibility.circom"
R1CS="${CIRCUIT%.circom}.r1cs"
WASM="${CIRCUIT%.circom}.wasm"
SYMBOLS="${CIRCUIT%.circom}.sym"
ZKEY="eligibility_final.zkey"
VK="verification_key.json"
PTAU="powersOfTau28_hez_final_10.ptau"  # Download if needed: wget https://hermez.s3-eu-west-1.amazonaws.com/powersOfTau28_hez_final_10.ptau

echo "Compiling $CIRCUIT..."

# Compile circuit
circom $CIRCUIT --r1cs --wasm --sym

# Setup phase 1
snarkjs groth16 setup $R1CS $PTAU ${CIRCUIT%.circom}_0000.zkey

# Phase 2 (contribute; in prod, use ceremony)
snarkjs zkey contribute ${CIRCUIT%.circom}_0000.zkey $ZKEY -n="First contribution" -v

# Export verification key
snarkjs groth16 export-verification-key $ZKEY $VK

echo "Compilation complete. Artifacts: $R1CS, $WASM, $ZKEY, $VK"
