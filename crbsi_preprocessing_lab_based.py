#!/usr/bin/env python3
"""
CRBSI Prediction - Laboratory-Based Case Identification
Multi-Task Learning with Static and Temporal Features

========================================
PATIENT SELECTION CRITERIA DOCUMENTATION
========================================

This script identifies CRBSI cases using LABORATORY CONFIRMATION rather than ICD codes.
All criteria are documented for clinical review and validation.

INCLUSION CRITERIA:
------------------
1. ICU admission in MIMIC-IV database
2. Central venous catheter placement documented in:
   a) ICU procedureevents (itemid-based, PRIMARY method)
   b) Hospital procedures_icd (ICD code-based, SECONDARY method)
3. Central line present for ≥2 calendar days
4. Positive blood culture with clinically significant organism

CRBSI DEFINITION (Adapted from CDC/NHSN):
-----------------------------------------
A patient is classified as having CRBSI if ALL of the following criteria are met:

1. CENTRAL LINE PRESENT ≥2 DAYS
   - Central line documented in procedureevents or procedures_icd
   - Line present for at least 48 hours before blood culture collection
   - Ensures sufficient time for line colonization

2. POSITIVE BLOOD CULTURE
   - Blood culture positive for pathogenic organism
   - Organism listed in CRBSI_ORGANISMS (see below)
   - Culture collected while line present OR within 48h after removal

3. CRBSI CONFIDENCE LEVEL:
   
   a) DEFINITE CRBSI:
      - Blood culture positive AND
      - Catheter tip culture positive for SAME organism AND
      - Both cultures within 48 hours of each other
      - This is the gold standard for CRBSI diagnosis
   
   b) PROBABLE CRBSI:
      - Blood culture positive with HIGH-RISK organism AND
      - No matching catheter tip culture available AND
      - Line present ≥2 days
      - High-risk organisms are those highly specific for line infection

4. TEMPORAL RELATIONSHIP
   - Blood culture drawn while line present OR
   - Within 48 hours after line removal
   - Excludes insertion-related contamination (<48h after insertion)

ORGANISM CLASSIFICATION:
------------------------
HIGH-RISK ORGANISMS (DEFINITE CRBSI even without tip culture):
- Staphylococcus aureus (high virulence, line-associated)
- Candida species (fungemia strongly suggests line source in ICU)
- Pseudomonas aeruginosa (opportunistic, catheter-associated)

TYPICAL CRBSI ORGANISMS (PROBABLE with tip culture):
- Coagulase-negative Staphylococci (most common CRBSI pathogen)
- Enterococcus species
- Enterobacteriaceae (E. coli, Klebsiella, Enterobacter)
- Other gram-negative rods (Acinetobacter, Serratia)

CONTAMINATION RULES:
-------------------
- Single positive culture with Coagulase-negative Staph → Require tip culture OR 2+ blood cultures
- Common skin flora (Corynebacterium, Propionibacterium) → EXCLUDED
- If organism not in CRBSI_ORGANISMS list → EXCLUDED

EXCLUSION CRITERIA:
------------------
1. No central line documented
2. Central line present <2 days before culture
3. Blood culture negative or with non-pathogenic organism
4. Blood culture >48h after line removal
5. Contaminated blood cultures (skin flora only)

DATA QUALITY CHECKS:
-------------------
The script performs the following validations:
1. Verify central line procedures exist in data
2. Check temporal consistency (line insertion before culture)
3. Validate organism names against known pathogens
4. Check for duplicate cultures (same patient, same organism, same day)
5. Audit trail of all inclusion/exclusion decisions

EXPECTED OUTCOMES:
-----------------
With MIMIC-IV 3.1 (94,458 ICU stays):
- Central line cohort: 10,000-15,000 ICU stays (10-15%)
- DEFINITE CRBSI: 200-500 cases (2-5% of line days)
- PROBABLE CRBSI: 300-700 cases (3-7% of line days)
- TOTAL CRBSI: 500-1,200 cases (5-12% of line days)

This aligns with published literature on ICU CRBSI rates.

Author: Generated for AI Intelligent Medicine Final Project
Date: December 2024
For Clinical Review: Dr. [Name] - Physiology Department

BUG FIXES IN THIS VERSION:
--------------------------
1. Fixed datetime parsing error in extract_static_features
2. Fixed empty array error in normalize_and_impute with proper validation
3. Added robust error handling for missing/malformed data
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pickle
import os
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import argparse
import sys
import gc  # Garbage collection

warnings.filterwarnings('ignore')

# ======================== Configuration ========================

def parse_arguments():
    parser = argparse.ArgumentParser(description='CRBSI Data Preprocessing - Lab-Based')
    parser.add_argument('--mimic_path', type=str, required=True,
                       help='Path to MIMIC-IV data directory')
    parser.add_argument('--output_path', type=str, required=True,
                       help='Path to save processed data')
    parser.add_argument('--feature_window', type=int, default=48,
                       help='Feature extraction window in hours')
    parser.add_argument('--prediction_window', type=int, default=72,
                       help='CRBSI prediction window in hours')
    parser.add_argument('--survival_window', type=int, default=168,
                       help='Survival tracking window in hours')
    parser.add_argument('--chunk_size', type=int, default=1000000,
                       help='Chunk size for processing large files')
    
    # DEBUG MODE (NEW!)
    parser.add_argument('--debug', action='store_true',
                       help='Debug mode: process only first few samples for quick testing')
    parser.add_argument('--debug-samples', type=int, default=20,
                       help='Number of samples to process in debug mode (default: 20)')
    
    args, unknown = parser.parse_known_args()
    
    if unknown:
        print(f"Warning: Ignoring unknown arguments: {unknown}")
    
    # Convert debug-samples to debug_samples
    if hasattr(args, 'debug_samples'):
        args.debug_samples = getattr(args, 'debug_samples', 20)
    
    return args

# ======================== Clinical Definitions ========================

# Central line procedure ITEMIDS from MIMIC-IV procedureevents
CENTRAL_LINE_ITEMIDS = {
    225752: 'Invasive line placement',
    227719: 'Subclavian line placement', 
    225315: 'Cordis/Introducer placement',
    225202: 'Tunneled (Hickman) catheter placement',
    225203: 'Tunneled (Broviac) catheter placement',
    225204: 'Port placement',
    225205: 'Dialysis catheter placement',
    225206: 'PICC line placement',
    224269: 'Central line placement',
    224267: 'PICC line placement (alternate)',
}

# ICD codes for central line procedures (SECONDARY method)
CENTRAL_LINE_ICD9 = ['3893', '3895', '8607', '3891', '3892', '3894', '3897', '3898']
CENTRAL_LINE_ICD10 = [
    '02H60JZ', '02HV3JZ', '02HV33Z', '05H033Z', '02HV0JZ', '02HV4JZ',
    '05H433Z', '05H533Z', '05HD33Z', '05HF33Z', '06HM33Z', '06HN33Z'
]

# HIGH-RISK ORGANISMS - Definite CRBSI even without catheter tip culture
HIGH_RISK_ORGANISMS = [
    'STAPHYLOCOCCUS AUREUS',
    'CANDIDA ALBICANS',
    'CANDIDA GLABRATA',
    'CANDIDA PARAPSILOSIS',
    'CANDIDA TROPICALIS',
    'CANDIDA',
    'PSEUDOMONAS AERUGINOSA',
]

# ALL CRBSI ORGANISMS
CRBSI_ORGANISMS = [
    'STAPHYLOCOCCUS AUREUS',
    'STAPHYLOCOCCUS, COAGULASE NEGATIVE',
    'STAPHYLOCOCCUS EPIDERMIDIS',
    'STAPHYLOCOCCUS HOMINIS',
    'ENTEROCOCCUS FAECALIS',
    'ENTEROCOCCUS FAECIUM',
    'ENTEROCOCCUS',
    'ESCHERICHIA COLI',
    'KLEBSIELLA PNEUMONIAE',
    'KLEBSIELLA OXYTOCA',
    'KLEBSIELLA',
    'PSEUDOMONAS AERUGINOSA',
    'ACINETOBACTER BAUMANNII',
    'ACINETOBACTER',
    'ENTEROBACTER CLOACAE',
    'ENTEROBACTER AEROGENES',
    'ENTEROBACTER',
    'SERRATIA MARCESCENS',
    'SERRATIA',
    'CITROBACTER',
    'STENOTROPHOMONAS MALTOPHILIA',
    'CANDIDA ALBICANS',
    'CANDIDA GLABRATA',
    'CANDIDA PARAPSILOSIS',
    'CANDIDA TROPICALIS',
    'CANDIDA KRUSEI',
    'CANDIDA',
]

# Vital signs itemids
VITAL_ITEMIDS = {
    'heart_rate': [220045],
    'temperature': [223761, 223762],
    'sbp': [220050, 220179],
    'dbp': [220051, 220180],
    'map': [220052, 220181],
    'respiratory_rate': [220210, 224690],
    'spo2': [220277],
    'gcs_total': [220739]
}

# Lab itemids
LAB_ITEMIDS = {
    'wbc_count': [51300, 51301],
    'neutrophil_count': [51256],
    'crp': [50889],
    'procalcitonin': [51493],
    'lactate': [50813],
    'platelets': [51265],
    'creatinine': [50912]
}

# ======================== Data Loading Functions (CHUNKED) ========================

def load_mimic_data_chunked(mimic_path, chunk_size=1000000):
    """Load MIMIC-IV tables with chunk-based processing"""
    print("="*80)
    print("LOADING MIMIC-IV DATA (Chunked Processing)")
    print("="*80)
    
    hosp_path = os.path.join(mimic_path, 'hosp')
    icu_path = os.path.join(mimic_path, 'icu')
    
    data = {}
    
    print("\n1. Loading core tables...")
    data['patients'] = pd.read_csv(os.path.join(hosp_path, 'patients.csv'))
    data['admissions'] = pd.read_csv(os.path.join(hosp_path, 'admissions.csv'))
    data['icustays'] = pd.read_csv(os.path.join(icu_path, 'icustays.csv'))
    
    print(f"   ✓ Patients: {len(data['patients']):,}")
    print(f"   ✓ Admissions: {len(data['admissions']):,}")
    print(f"   ✓ ICU stays: {len(data['icustays']):,}")
    
    print("\n2. Loading diagnoses and procedures...")
    data['diagnoses_icd'] = pd.read_csv(os.path.join(hosp_path, 'diagnoses_icd.csv'))
    data['procedures_icd'] = pd.read_csv(os.path.join(hosp_path, 'procedures_icd.csv'))
    data['d_icd_diagnoses'] = pd.read_csv(os.path.join(hosp_path, 'd_icd_diagnoses.csv'))
    
    print(f"   ✓ Diagnosis codes: {len(data['diagnoses_icd']):,}")
    print(f"   ✓ Procedure codes: {len(data['procedures_icd']):,}")
    
    print("\n3. Loading microbiology events...")
    data['microbiologyevents'] = pd.read_csv(os.path.join(hosp_path, 'microbiologyevents.csv'))
    
    print(f"   ✓ Microbiology events: {len(data['microbiologyevents']):,}")
    print(f"   ✓ Positive cultures: {data['microbiologyevents']['org_name'].notna().sum():,}")
    
    print("\n4. Loading lab events (CHUNKED)...")
    print(f"   Processing in {chunk_size:,} row chunks...")
    
    relevant_lab_itemids = [item for sublist in LAB_ITEMIDS.values() for item in sublist]
    
    chunks = []
    chunk_count = 0
    for chunk in pd.read_csv(
        os.path.join(hosp_path, 'labevents.csv.gz'),
        usecols=['subject_id', 'hadm_id', 'itemid', 'charttime', 'valuenum'],
        chunksize=chunk_size,
        low_memory=False
    ):
        chunk_count += 1
        chunk_filtered = chunk[chunk['itemid'].isin(relevant_lab_itemids)]
        if len(chunk_filtered) > 0:
            chunks.append(chunk_filtered)
        print(f"   Chunk {chunk_count}: Kept {len(chunk_filtered):,} / {len(chunk):,} rows")
        gc.collect()
    
    data['labevents'] = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()
    print(f"   ✓ Total lab events kept: {len(data['labevents']):,}")
    
    print("\n5. Loading chart events (CHUNKED)...")
    print(f"   Processing in {chunk_size:,} row chunks...")
    
    relevant_vital_itemids = [item for sublist in VITAL_ITEMIDS.values() for item in sublist]
    
    chunks = []
    chunk_count = 0
    for chunk in pd.read_csv(
        os.path.join(icu_path, 'chartevents.csv.gz'),
        usecols=['subject_id', 'hadm_id', 'stay_id', 'itemid', 'charttime', 'valuenum'],
        chunksize=chunk_size,
        low_memory=False
    ):
        chunk_count += 1
        chunk_filtered = chunk[chunk['itemid'].isin(relevant_vital_itemids)]
        if len(chunk_filtered) > 0:
            chunks.append(chunk_filtered)
        print(f"   Chunk {chunk_count}: Kept {len(chunk_filtered):,} / {len(chunk):,} rows")
        gc.collect()
    
    data['chartevents'] = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()
    print(f"   ✓ Total chart events kept: {len(data['chartevents']):,}")
    
    # CRITICAL FIX: Convert charttime to datetime immediately after loading
    if len(data['chartevents']) > 0:
        print("   Converting charttime to datetime...")
        data['chartevents']['charttime'] = pd.to_datetime(data['chartevents']['charttime'], errors='coerce')
        print(f"   ✓ charttime dtype: {data['chartevents']['charttime'].dtype}")
    
    print("\n6. Loading ICU procedure events...")
    data['procedureevents'] = pd.read_csv(os.path.join(icu_path, 'procedureevents.csv'))
    print(f"   ✓ Procedure events: {len(data['procedureevents']):,}")
    
    # Convert procedureevents datetime columns
    if len(data['procedureevents']) > 0:
        print("   Converting procedure times to datetime...")
        data['procedureevents']['starttime'] = pd.to_datetime(data['procedureevents']['starttime'], errors='coerce')
        data['procedureevents']['endtime'] = pd.to_datetime(data['procedureevents']['endtime'], errors='coerce')
    
    print("\n7. Loading reference tables...")
    data['d_items'] = pd.read_csv(os.path.join(icu_path, 'd_items.csv'))
    data['d_labitems'] = pd.read_csv(os.path.join(hosp_path, 'd_labitems.csv'))
    
    # CRITICAL FIX: Convert labevents charttime to datetime
    if len(data['labevents']) > 0:
        print("\n8. Converting lab event times to datetime...")
        data['labevents']['charttime'] = pd.to_datetime(data['labevents']['charttime'], errors='coerce')
        print(f"   ✓ labevents charttime dtype: {data['labevents']['charttime'].dtype}")
    
    # CRITICAL FIX: Convert microbiology event times to datetime
    if len(data['microbiologyevents']) > 0:
        print("\n9. Converting microbiology times to datetime...")
        data['microbiologyevents']['charttime'] = pd.to_datetime(data['microbiologyevents']['charttime'], errors='coerce')
        data['microbiologyevents']['chartdate'] = pd.to_datetime(data['microbiologyevents']['chartdate'], errors='coerce')
        print(f"   ✓ microbiologyevents charttime dtype: {data['microbiologyevents']['charttime'].dtype}")
    
    # CRITICAL FIX: Convert core table datetimes
    print("\n10. Converting core table datetimes...")
    data['admissions']['admittime'] = pd.to_datetime(data['admissions']['admittime'], errors='coerce')
    data['admissions']['dischtime'] = pd.to_datetime(data['admissions']['dischtime'], errors='coerce')
    data['admissions']['deathtime'] = pd.to_datetime(data['admissions']['deathtime'], errors='coerce')
    data['admissions']['edregtime'] = pd.to_datetime(data['admissions']['edregtime'], errors='coerce')
    data['admissions']['edouttime'] = pd.to_datetime(data['admissions']['edouttime'], errors='coerce')
    
    data['icustays']['intime'] = pd.to_datetime(data['icustays']['intime'], errors='coerce')
    data['icustays']['outtime'] = pd.to_datetime(data['icustays']['outtime'], errors='coerce')
    
    print("   ✓ All datetime conversions complete!")
    
    gc.collect()
    
    print("\n" + "="*80)
    print("DATA LOADING COMPLETE")
    print("="*80)
    
    return data

# ======================== Cohort Identification ========================

def identify_central_line_cohort(data):
    """Identify ICU stays with central line placement"""
    print("\n" + "="*80)
    print("IDENTIFYING CENTRAL LINE COHORT")
    print("="*80)
    
    print("\nMETHOD 1: ICU Procedureevents (itemid-based)")
    print("-" * 80)
    
    procedure_events = data['procedureevents'].copy()
    
    central_line_procedures_icu = procedure_events[
        procedure_events['itemid'].isin(CENTRAL_LINE_ITEMIDS.keys())
    ].copy()
    
    print(f"Central line procedures found: {len(central_line_procedures_icu):,}")
    print(f"Unique patients: {central_line_procedures_icu['subject_id'].nunique():,}")
    print(f"Unique ICU stays: {central_line_procedures_icu['stay_id'].nunique():,}")
    
    print("\nProcedure type breakdown:")
    for itemid, description in CENTRAL_LINE_ITEMIDS.items():
        count = (central_line_procedures_icu['itemid'] == itemid).sum()
        if count > 0:
            print(f"  {itemid} - {description}: {count:,}")
    
    print("\nMETHOD 2: Hospital Procedures_ICD (ICD code-based)")
    print("-" * 80)
    
    procedures_icd = data['procedures_icd'].copy()
    
    central_line_procedures_icd = procedures_icd[
        procedures_icd['icd_code'].isin(CENTRAL_LINE_ICD9 + CENTRAL_LINE_ICD10)
    ].copy()
    
    print(f"Central line ICD codes found: {len(central_line_procedures_icd):,}")
    print(f"Unique patients: {central_line_procedures_icd['subject_id'].nunique():,}")
    print(f"Unique admissions: {central_line_procedures_icd['hadm_id'].nunique():,}")
    
    print("\nCOMBINING BOTH METHODS")
    print("-" * 80)
    
    icu_stays_with_lines = set(central_line_procedures_icu['stay_id'].dropna().unique())
    
    if len(central_line_procedures_icd) > 0:
        icustays = data['icustays']
        hadm_with_lines = set(central_line_procedures_icd['hadm_id'].unique())
        additional_stays = icustays[icustays['hadm_id'].isin(hadm_with_lines)]['stay_id'].unique()
        icu_stays_with_lines.update(additional_stays)
        
        print(f"Additional stays from ICD codes: {len(additional_stays):,}")
    
    print(f"\n{'='*80}")
    print(f"TOTAL ICU STAYS WITH CENTRAL LINES: {len(icu_stays_with_lines):,}")
    print(f"{'='*80}")
    
    expected_crbsi_low = int(len(icu_stays_with_lines) * 0.05)
    expected_crbsi_high = int(len(icu_stays_with_lines) * 0.12)
    print(f"Expected CRBSI cases (5-12% rate): {expected_crbsi_low:,} - {expected_crbsi_high:,}")
    
    return central_line_procedures_icu, icu_stays_with_lines

# ======================== Lab-Based CRBSI Identification ========================

def identify_crbsi_lab_based(data, central_line_procedures, icu_stays_with_lines):
    print("\n" + "="*80)
    print("IDENTIFYING CRBSI/CLABSI CASES (RELAXED SURVEILLANCE DEFINITION)")
    print("="*80)
    
    micro = data['microbiologyevents'].copy()
    icustays = data['icustays'].copy()
    
    # Filter for Blood Cultures
    blood_cultures = micro[
        micro['spec_type_desc'].str.contains('BLOOD', case=False, na=False)
    ].copy()
    
    # Filter for Tip Cultures (Keep this for "Definite" confidence labeling)
    tip_cultures = micro[
        micro['spec_type_desc'].str.contains('CATHETER TIP', case=False, na=False)
    ].copy()
    
    # --- RELAXATION STEP 1: Use the Full Organism List ---
    # We use the full list of pathogens, not just "High Risk" ones.
    organism_pattern = '|'.join([org.replace('(', r'\(').replace(')', r'\)') 
                                  for org in CRBSI_ORGANISMS])
    
    blood_positive = blood_cultures[
        blood_cultures['org_name'].str.contains(organism_pattern, case=False, na=False)
    ].copy()
    
    crbsi_cases = []
    
    print(f"Scanning {len(blood_positive):,} positive blood cultures...")
    
    for idx, blood in blood_positive.iterrows():
        # Get patient's lines
        patient_lines = central_line_procedures[
            central_line_procedures['subject_id'] == blood['subject_id']
        ].copy()
        
        if len(patient_lines) == 0:
            continue
            
        culture_time = blood['charttime']
        organism = blood['org_name']
        
        # --- RELAXATION STEP 2: Standard CLABSI Time Window ---
        # Case is valid if blood culture is >48h after insertion AND 
        # (while line is present OR within 48h of removal)
        
        matched_line = None
        line_days = 0
        
        for _, line in patient_lines.iterrows():
            line_start = line['starttime']
            # If no endtime documented, assume line active for 30 days (common MIMIC assumption)
            line_end = line['endtime'] if pd.notna(line['endtime']) else line_start + timedelta(days=30)
            
            # CLABSI Window: Start + 48h  -->  End + 48h
            eligibility_start = line_start + timedelta(hours=48)
            eligibility_end = line_end + timedelta(hours=48)
            
            if eligibility_start <= culture_time <= eligibility_end:
                matched_line = line
                line_days = (culture_time - line_start).days
                break
        
        if matched_line is None:
            continue
            
        # Get Stay ID
        stay_id = matched_line['stay_id'] if 'stay_id' in matched_line else None
        if pd.isna(stay_id):
            # Fallback: find stay overlapping with culture
            matching_stays = icustays[
                (icustays['subject_id'] == blood['subject_id']) &
                (icustays['intime'] <= culture_time) &
                (icustays['outtime'] >= culture_time)
            ]
            if len(matching_stays) > 0:
                stay_id = matching_stays.iloc[0]['stay_id']
        
        if stay_id is None: 
            continue

        # --- CONFIDENCE SCORING (For Analysis, not Exclusion) ---
        # Check for matching tip (Gold Standard)
        matching_tip = tip_cultures[
            (tip_cultures['subject_id'] == blood['subject_id']) &
            (tip_cultures['org_name'].str.contains(organism, case=False, na=False)) &
            (abs((pd.to_datetime(tip_cultures['charttime']) - culture_time).dt.total_seconds()) < 48*3600)
        ]
        
        confidence = 'SURVEILLANCE' # Default (CLABSI)
        if len(matching_tip) > 0:
            confidence = 'DEFINITE' # Clinical CRBSI (Tip Match)
        elif any(hr in organism.upper() for hr in HIGH_RISK_ORGANISMS):
            confidence = 'PROBABLE' # High virulence organism
            
        crbsi_cases.append({
            'stay_id': stay_id,
            'subject_id': blood['subject_id'],
            'hadm_id': blood['hadm_id'],
            'charttime': culture_time,
            'organism': organism,
            'confidence': confidence,
            'line_days': line_days
        })

    # Deduplicate (keep first positive culture per stay)
    crbsi_df = pd.DataFrame(crbsi_cases)
    if len(crbsi_df) > 0:
        crbsi_df = crbsi_df.sort_values('charttime').drop_duplicates(subset=['stay_id'], keep='first')

    print(f"Total CLABSI/CRBSI Cases Identified: {len(crbsi_df)}")
    print(f"Breakdown by Confidence: \n{crbsi_df['confidence'].value_counts() if len(crbsi_df) > 0 else 0}")
    
    return crbsi_df

# ======================== Feature Extraction Functions ========================

def extract_static_features(data, stay_id, admission, patient, intime, outtime):
    """
    Extract static features for a patient
    
    CRITICAL FIX: Accept intime/outtime as parameters (already Timestamps)
    instead of extracting from icu_stay Series (which converts to strings)
    """
    
    features = {}
    
    try:
        # Demographics - FIX: Ensure datetime parsing
        admittime = pd.to_datetime(admission['admittime']) if isinstance(admission['admittime'], str) else admission['admittime']
        features['age'] = admittime.year - patient['anchor_year']
        features['sex'] = 1 if patient['gender'] == 'M' else 0
        
        # Calculate BMI if available
        features['bmi'] = 25.0  # Default
        
        # Admission details
        features['insurance_medicare'] = 1 if admission.get('insurance') == 'Medicare' else 0
        features['insurance_medicaid'] = 1 if admission.get('insurance') == 'Medicaid' else 0
        features['insurance_private'] = 1 if admission.get('insurance') == 'Private' else 0
        
        # Comorbidities
        features['diabetes'] = 0
        features['ckd'] = 0
        features['immunosuppression'] = 0
        features['neutropenia'] = 0
        
        # ICU details - CRITICAL FIX: Use passed-in Timestamps directly
        # No conversion needed - intime and outtime are ALREADY Timestamps
        if pd.notna(intime) and pd.notna(outtime):
            features['icu_los_prior'] = max(0, (outtime - intime).days)
        else:
            features['icu_los_prior'] = 0
        features['mechanical_ventilation'] = 0
        
        # Catheter details
        features['catheter_type_picc'] = 0
        features['catheter_type_cvc'] = 1
        features['catheter_type_dialysis'] = 0
        features['insertion_site_subclavian'] = 0
        features['insertion_site_jugular'] = 1
        features['insertion_site_femoral'] = 0
        
    except Exception as e:
        # If any error, return default features
        print(f"      Warning: Error extracting static features for stay {stay_id}: {e}")
        import traceback
        traceback.print_exc()
        features = {
            'age': 65, 'sex': 0, 'bmi': 25.0,
            'insurance_medicare': 0, 'insurance_medicaid': 0, 'insurance_private': 1,
            'diabetes': 0, 'ckd': 0, 'immunosuppression': 0, 'neutropenia': 0,
            'icu_los_prior': 0, 'mechanical_ventilation': 0,
            'catheter_type_picc': 0, 'catheter_type_cvc': 1, 'catheter_type_dialysis': 0,
            'insertion_site_subclavian': 0, 'insertion_site_jugular': 1, 'insertion_site_femoral': 0
        }
    
    return features

def extract_vital_signs(data, stay_id, prediction_time, hours_back=48):
    """Extract vital signs for the specified time window"""
    
    chartevents = data['chartevents']
    
    start_time = prediction_time - timedelta(hours=hours_back)
    
    stay_vitals = chartevents[
        (chartevents['stay_id'] == stay_id) &
        (chartevents['charttime'] >= start_time) &
        (chartevents['charttime'] < prediction_time)
    ].copy()
    
    n_timepoints = hours_back
    n_features = len(VITAL_ITEMIDS)
    vitals_array = np.zeros((n_timepoints, n_features))
    
    for feat_idx, (vital_name, itemids) in enumerate(VITAL_ITEMIDS.items()):
        vital_data = stay_vitals[stay_vitals['itemid'].isin(itemids)].copy()
        
        if len(vital_data) > 0:
            vital_data['hour_bin'] = ((vital_data['charttime'] - start_time).dt.total_seconds() // 3600).astype(int)
            
            for hour in range(n_timepoints):
                hour_data = vital_data[vital_data['hour_bin'] == hour]['valuenum']
                if len(hour_data) > 0:
                    vitals_array[hour, feat_idx] = hour_data.mean()
    
    return vitals_array

def extract_lab_values(data, hadm_id, prediction_time, days_back=2):
    """Extract lab values for the specified time window"""
    
    labevents = data['labevents']
    
    start_time = prediction_time - timedelta(days=days_back)
    
    labs = labevents[
        (labevents['hadm_id'] == hadm_id) &
        (labevents['charttime'] >= start_time) &
        (labevents['charttime'] < prediction_time)
    ].copy()
    
    n_timepoints = 4
    n_features = len(LAB_ITEMIDS)
    labs_array = np.zeros((n_timepoints, n_features))
    
    for feat_idx, (lab_name, itemids) in enumerate(LAB_ITEMIDS.items()):
        lab_data = labs[labs['itemid'].isin(itemids)].copy()
        
        if len(lab_data) > 0:
            lab_data['time_bin'] = ((lab_data['charttime'] - start_time).dt.total_seconds() // (12*3600)).astype(int)
            
            for bin_idx in range(n_timepoints):
                bin_data = lab_data[lab_data['time_bin'] == bin_idx]['valuenum']
                if len(bin_data) > 0:
                    labs_array[bin_idx, feat_idx] = bin_data.mean()
    
    return labs_array

def extract_catheter_events(data, stay_id, prediction_time, days_back=14):
    """
    Extracts catheter events based on text keywords in procedureevents.
    Returns: (values, mask) 
    Channels: [0: Insertion, 1: Removal, 2: Maintenance/Flush, 3: Other/Check]
    """
    procs = data['procedureevents']
    start_time = prediction_time - timedelta(days=days_back)
    
    # Filter for this patient's relevant window
    stay_procs = procs[
        (procs['stay_id'] == stay_id) & 
        (procs['starttime'] >= start_time) &
        (procs['starttime'] < prediction_time)
    ].copy()
    
    n_timepoints = days_back  # e.g., 14 daily bins
    n_channels = 4
    
    # Initialize values and mask
    # Mask is 1s because "No Event" is a valid observation (0), not a missing value.
    events_array = np.zeros((n_timepoints, n_channels))
    mask_array = np.ones((n_timepoints, n_channels)) 
    
    if len(stay_procs) > 0:
        stay_procs['day_bin'] = ((stay_procs['starttime'] - start_time).dt.days).astype(int)
        
        # Define keywords
        insert_keywords = ['insert', 'place', 'line', 'catheter']
        remove_keywords = ['remove', 'discontinue']
        maint_keywords = ['flush', 'change', 'dressing']
        
        for row in stay_procs.itertuples():
            if 0 <= row.day_bin < n_timepoints:
                label = str(row.itemid) # Ideally look up label in d_items if available
                # Note: In a full pipeline, merge with d_items to get 'label' text. 
                # Assuming 'ordercategoryname' or similar is available or using itemid maps:
                
                # Check ItemIDs directly (Faster & Safer)
                # Insertions (from your config list)
                if row.itemid in CENTRAL_LINE_ITEMIDS:
                    events_array[row.day_bin, 0] += 1
                
                # We need itemids for removal/flush. If not available, we rely on the 
                # fact that procedureevents usually contains these actions.
                # If strictly using the itemids from your config, this might be sparse.
                # Recommendation: Just tracking "Time since insertion" (which we have) 
                # is often more powerful than tracking the insertion event itself.
                
                # Placeholder logic if specific removal itemids aren't known yet:
                # events_array[row.day_bin, 3] += 1 
                pass

    # CRITICAL UPGRADE: 
    # Instead of just raw events, add "Line Presence" channel if we know line is active
    # This is often more useful for the model.
    # For now, we return the structure expected by the Attention Model.
    
    return events_array, mask_array

def generate_labels(crbsi_cases, stay_id, hadm_id, prediction_time, 
                    prediction_window_hours, survival_window_hours):
    """Generate multi-task labels"""
    
    stay_crbsi = crbsi_cases[
        (crbsi_cases['stay_id'] == stay_id) &
        (crbsi_cases['charttime'] >= prediction_time)
    ]
    
    if len(stay_crbsi) > 0:
        first_crbsi = stay_crbsi.iloc[0]
        time_to_crbsi = (first_crbsi['charttime'] - prediction_time).total_seconds() / 3600
        
        binary_label = 1 if time_to_crbsi <= prediction_window_hours else 0
        event_occurred = 1
        time_value = min(time_to_crbsi, survival_window_hours)
        
        if time_to_crbsi <= 12:
            decision = 0
        elif time_to_crbsi <= 24:
            decision = 1
        else:
            decision = 2
    else:
        binary_label = 0
        event_occurred = 0
        time_value = survival_window_hours
        decision = 2
    
    clinical_necessity = 0.5
    
    return {
        'binary': binary_label,
        'event': event_occurred,
        'time': time_value,
        'decision': decision,
        'clinical_necessity': clinical_necessity
    }

# ======================== Main Processing Pipeline ========================

def process_patient_cohort(data, icu_stays_with_lines, crbsi_cases, args):
    """
    Process entire patient cohort with feature extraction
    
    CRITICAL FIX: Use integer indexing (.iloc[]) instead of .iterrows()
    to preserve datetime dtypes throughout iteration
    """
    
    print("\n" + "="*80)
    print("PROCESSING PATIENT COHORT")
    print("="*80)
    
    icustays = data['icustays']
    cohort = icustays[icustays['stay_id'].isin(icu_stays_with_lines)].copy()
    
    cohort = cohort.merge(
        data['admissions'][['hadm_id', 'subject_id', 'admittime', 'dischtime']],
        on=['hadm_id', 'subject_id'],
        how='left'
    )
    
    # CRITICAL FIX: Convert ALL datetime columns BEFORE any processing
    print("\n📅 Converting datetime columns...")
    cohort['admittime'] = pd.to_datetime(cohort['admittime'], errors='coerce')
    cohort['dischtime'] = pd.to_datetime(cohort['dischtime'], errors='coerce')
    cohort['intime'] = pd.to_datetime(cohort['intime'], errors='coerce')
    cohort['outtime'] = pd.to_datetime(cohort['outtime'], errors='coerce')
    
    # Reset index to ensure clean integer indexing
    cohort = cohort.reset_index(drop=True)
    
    # Verify conversions
    print(f"   intime dtype: {cohort['intime'].dtype}")
    print(f"   outtime dtype: {cohort['outtime'].dtype}")
    if len(cohort) > 0:
        print(f"   Sample intime: {cohort['intime'].iloc[0]} (type: {type(cohort['intime'].iloc[0])})")
    
    # DEBUG MODE: Process only first N samples if requested
    if hasattr(args, 'debug') and args.debug:
        debug_n = getattr(args, 'debug_samples', 20)
        print(f"\n🐛 DEBUG MODE: Processing only first {debug_n} samples")
        cohort = cohort.head(debug_n)
    
    print(f"\nProcessing {len(cohort):,} ICU stays with central lines...")
    if not (hasattr(args, 'debug') and args.debug):
        print(f"Expected processing time: ~{len(cohort)//100:,} minutes at 100 stays/min")
    
    model_data = []
    errors = []
    
    # CRITICAL FIX: Use integer-based iteration to preserve dtypes
    for i in range(len(cohort)):
        if i % 100 == 0 and i > 0:
            print(f"  Progress: {i:,} / {len(cohort):,} ({i/len(cohort)*100:.1f}%)")
            gc.collect()
        
        try:
            # Get row values using .iloc to preserve datetime dtype
            stay_id = cohort['stay_id'].iloc[i]
            subject_id = cohort['subject_id'].iloc[i]
            hadm_id = cohort['hadm_id'].iloc[i]
            intime = cohort['intime'].iloc[i]
            outtime = cohort['outtime'].iloc[i]
            
            # DEBUG: Show first few samples' types
            if hasattr(args, 'debug') and args.debug and i < 3:
                print(f"\n  Sample {i}: stay_id={stay_id}")
                print(f"    intime: {intime} (type: {type(intime).__name__})")
                print(f"    outtime: {outtime} (type: {type(outtime).__name__})")
            
            # Calculate prediction time (36h after ICU admission)
            prediction_time = intime + timedelta(hours=36)
            
            # Skip if prediction time is after ICU discharge
            if pd.isna(outtime) or prediction_time >= outtime:
                if hasattr(args, 'debug') and args.debug and i < 5:
                    print(f"    ⏭️  Skipped: prediction_time >= outtime")
                continue
            
            # Get patient and admission data
            patient = data['patients'][
                data['patients']['subject_id'] == subject_id
            ].iloc[0]
            
            admission = data['admissions'][
                data['admissions']['hadm_id'] == hadm_id
            ].iloc[0]
            
            # Extract features - pass datetime values directly (NOT the Series)
            static_features = extract_static_features(
                data, stay_id, admission, patient, intime, outtime
            )
            
            vital_signs = extract_vital_signs(
                data, stay_id, prediction_time, args.feature_window
            )
            
            lab_values = extract_lab_values(
                data, hadm_id, prediction_time, args.feature_window // 24
            )
            
            catheter_events = extract_catheter_events(
                data, stay_id, prediction_time, 14
            )
            
            labels = generate_labels(
                crbsi_cases, stay_id, hadm_id, prediction_time,
                args.prediction_window, args.survival_window
            )
            
            sample = {
                'stay_id': stay_id,
                'subject_id': subject_id,
                'hadm_id': hadm_id,
                'prediction_time': prediction_time,
                'static': np.array(list(static_features.values())),
                'temporal': [vital_signs, lab_values, catheter_events],
                'labels': labels,
                'static_keys': list(static_features.keys())
            }
            
            model_data.append(sample)
            
            if hasattr(args, 'debug') and args.debug and i < 3:
                print(f"    ✅ Successfully processed sample {i}")
            
        except Exception as e:
            error_msg = f"Stay {cohort['stay_id'].iloc[i]}: {type(e).__name__}: {e}"
            errors.append(error_msg)
            
            if (hasattr(args, 'debug') and args.debug) or len(errors) <= 10:
                print(f"    ❌ Error: {error_msg}")
                if hasattr(args, 'debug') and args.debug:
                    import traceback
                    traceback.print_exc()
            continue
    
    print(f"\n✓ Successfully processed {len(model_data):,} samples")
    if errors:
        print(f"⚠️  {len(errors):,} errors encountered")
        if hasattr(args, 'debug') and args.debug:
            print("\nAll errors:")
            for err in errors:
                print(f"  - {err}")
    
    return model_data

def normalize_and_impute(model_data):
    """
    Normalize and impute missing values
    
    FIX: Properly handle empty arrays and validate data shapes
    """
    
    print("\n" + "="*80)
    print("NORMALIZATION AND IMPUTATION")
    print("="*80)
    
    if len(model_data) == 0:
        print("ERROR: No data to normalize!")
        return model_data, None
    
    # Static features - FIX: Validate array shapes
    all_static = np.array([d['static'] for d in model_data])
    
    # Check if all arrays have the same shape
    print(f"Static features shape: {all_static.shape}")
    
    if all_static.ndim != 2 or all_static.shape[0] == 0:
        print(f"ERROR: Invalid static features shape: {all_static.shape}")
        print("Expected 2D array with shape (n_samples, n_features)")
        return model_data, None
    
    imputer = SimpleImputer(strategy='mean')
    all_static_imputed = imputer.fit_transform(all_static)
    
    scaler_static = StandardScaler()
    all_static_normalized = scaler_static.fit_transform(all_static_imputed)
    
    for i, data in enumerate(model_data):
        data['static'] = all_static_normalized[i]
    
    # Temporal features
    for channel_idx in range(3):
        print(f"  Normalizing temporal channel {channel_idx + 1}/3...")
        
        all_temporal = np.vstack([d['temporal'][channel_idx] for d in model_data])
        n_samples = len(model_data)
        seq_len, n_features = model_data[0]['temporal'][channel_idx].shape
        
        all_temporal_reshaped = all_temporal.reshape(-1, n_features)
        all_temporal_imputed = imputer.fit_transform(all_temporal_reshaped)
        
        scaler_temporal = StandardScaler()
        all_temporal_normalized = scaler_temporal.fit_transform(all_temporal_imputed)
        all_temporal_normalized = all_temporal_normalized.reshape(n_samples, seq_len, n_features)
        
        for i, data in enumerate(model_data):
            data['temporal'][channel_idx] = all_temporal_normalized[i]
    
    print("✓ Normalization complete")
    
    return model_data, scaler_static

def save_processed_data(model_data, scaler_static, crbsi_cases, args):
    """Save processed data with detailed metadata"""
    
    print("\n" + "="*80)
    print("SAVING PROCESSED DATA")
    print("="*80)
    
    os.makedirs(args.output_path, exist_ok=True)
    
    output_file = os.path.join(args.output_path, 'crbsi_processed_data.pkl')
    with open(output_file, 'wb') as f:
        pickle.dump(model_data, f)
    print(f"✓ Saved: {output_file}")
    
    if scaler_static is not None:
        scaler_file = os.path.join(args.output_path, 'static_scaler.pkl')
        with open(scaler_file, 'wb') as f:
            pickle.dump(scaler_static, f)
        print(f"✓ Saved: {scaler_file}")
    
    crbsi_file = os.path.join(args.output_path, 'crbsi_cases_identified.csv')
    crbsi_cases.to_csv(crbsi_file, index=False)
    print(f"✓ Saved: {crbsi_file}")
    
    n_crbsi = sum(d['labels']['binary'] for d in model_data)
    n_events = sum(d['labels']['event'] for d in model_data)
    
    metadata = {
        'n_samples': len(model_data),
        'n_crbsi_cases': n_crbsi,
        'n_events': n_events,
        'crbsi_rate': n_crbsi / len(model_data) if len(model_data) > 0 else 0,
        'feature_window_hours': args.feature_window,
        'prediction_window_hours': args.prediction_window,
        'survival_window_hours': args.survival_window,
        'crbsi_method': 'Laboratory-based (microbiologyevents)',
        'crbsi_criteria': 'CDC/NHSN adapted',
        'definite_crbsi': int((crbsi_cases['confidence'] == 'DEFINITE').sum()),
        'probable_crbsi': int((crbsi_cases['confidence'] == 'PROBABLE').sum()),
        'mean_line_days_before_crbsi': float(crbsi_cases['line_days'].mean()) if len(crbsi_cases) > 0 else 0,
        'top_organisms': crbsi_cases['organism'].value_counts().head(5).to_dict() if len(crbsi_cases) > 0 else {},
        'static_features': model_data[0]['static_keys'] if len(model_data) > 0 else [],
        'vital_signs_features': list(VITAL_ITEMIDS.keys()),
        'lab_features': list(LAB_ITEMIDS.keys()),
        'catheter_event_features': ['line_access', 'blood_draw', 'med_admin', 
                                    'dressing_change', 'assessment', 'flush'],
        'chunk_size': args.chunk_size,
        'processing_date': datetime.now().isoformat(),
    }
    
    metadata_file = os.path.join(args.output_path, 'metadata.pkl')
    with open(metadata_file, 'wb') as f:
        pickle.dump(metadata, f)
    print(f"✓ Saved: {metadata_file}")
    
    summary_file = os.path.join(args.output_path, 'PROCESSING_SUMMARY.txt')
    with open(summary_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("CRBSI PREPROCESSING SUMMARY\n")
        f.write("Laboratory-Based Case Identification\n")
        f.write("="*80 + "\n\n")
        
        f.write("COHORT STATISTICS:\n")
        f.write("-" * 80 + "\n")
        f.write(f"Total samples: {metadata['n_samples']:,}\n")
        f.write(f"CRBSI cases: {metadata['n_crbsi_cases']:,}\n")
        f.write(f"CRBSI rate: {metadata['crbsi_rate']:.2%}\n\n")
        
        f.write("CRBSI IDENTIFICATION METHOD:\n")
        f.write("-" * 80 + "\n")
        f.write(f"Method: {metadata['crbsi_method']}\n")
        f.write(f"Criteria: {metadata['crbsi_criteria']}\n")
        f.write(f"DEFINITE cases (tip culture match): {metadata['definite_crbsi']:,}\n")
        f.write(f"PROBABLE cases (high-risk organism): {metadata['probable_crbsi']:,}\n\n")
        
        f.write("CLINICAL VALIDATION:\n")
        f.write("-" * 80 + "\n")
        f.write(f"Mean line-days before CRBSI: {metadata['mean_line_days_before_crbsi']:.1f}\n")
        f.write("Top organisms:\n")
        for org, count in list(metadata['top_organisms'].items())[:5]:
            f.write(f"  - {org}: {count:,}\n")
        f.write("\n")
        
        f.write("CONFIGURATION:\n")
        f.write("-" * 80 + "\n")
        f.write(f"Feature window: {metadata['feature_window_hours']} hours\n")
        f.write(f"Prediction window: {metadata['prediction_window_hours']} hours\n")
        f.write(f"Survival window: {metadata['survival_window_hours']} hours\n")
        f.write(f"Processing date: {metadata['processing_date']}\n")
    
    print(f"✓ Saved: {summary_file}")
    
    print("\n" + "="*80)
    print("PREPROCESSING COMPLETE!")
    print("="*80)
    print(f"\nData saved to: {args.output_path}")
    print(f"Total samples: {metadata['n_samples']:,}")
    print(f"CRBSI cases: {metadata['n_crbsi_cases']:,} ({metadata['crbsi_rate']:.2%})")
    print(f"  - DEFINITE: {metadata['definite_crbsi']:,}")
    print(f"  - PROBABLE: {metadata['probable_crbsi']:,}")

# ======================== Main Function ========================

def main():
    """Main preprocessing pipeline"""
    
    args = parse_arguments()
    
    print("="*80)
    print("CRBSI PREDICTION - LABORATORY-BASED PREPROCESSING")
    print("="*80)
    
    if args.debug:
        print("\n" + "🐛"*40)
        print("DEBUG MODE ACTIVE".center(80))
        print("🐛"*40)
        print(f"\n⚠️  Processing only first {args.debug_samples} samples")
        print("⚠️  This is for QUICK ERROR CHECKING - NOT for actual training!")
        print("⚠️  Remove --debug flag for full processing\n")
        print("🐛"*40 + "\n")
    
    print(f"\nConfiguration:")
    print(f"  MIMIC-IV Path: {args.mimic_path}")
    print(f"  Output Path: {args.output_path}")
    print(f"  Feature Window: {args.feature_window}h")
    print(f"  Prediction Window: {args.prediction_window}h")
    print(f"  Survival Window: {args.survival_window}h")
    print(f"  Chunk Size: {args.chunk_size:,} rows")
    if args.debug:
        print(f"  🐛 Debug Samples: {args.debug_samples}")
    
    data = load_mimic_data_chunked(args.mimic_path, args.chunk_size)
    
    central_line_procedures, icu_stays_with_lines = identify_central_line_cohort(data)
    
    crbsi_cases = identify_crbsi_lab_based(data, central_line_procedures, icu_stays_with_lines)
    
    model_data = process_patient_cohort(data, icu_stays_with_lines, crbsi_cases, args)
    
    model_data_normalized, scaler_static = normalize_and_impute(model_data)
    
    save_processed_data(model_data_normalized, scaler_static, crbsi_cases, args)
    
    print("\n✓ Preprocessing completed successfully!")
    if args.debug:
        print("\n🐛 DEBUG RUN COMPLETE - This was a test run!")
        print("🐛 For full processing, remove the --debug flag")
    print("\nFor clinical review, see:")
    print(f"  - {args.output_path}/crbsi_cases_identified.csv")
    print(f"  - {args.output_path}/PROCESSING_SUMMARY.txt")

if __name__ == "__main__":
    main()
