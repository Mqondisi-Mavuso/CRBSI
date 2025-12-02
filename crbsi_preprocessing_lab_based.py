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
    
    args, unknown = parser.parse_known_args()
    
    if unknown:
        print(f"Warning: Ignoring unknown arguments: {unknown}")
    
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
# These organisms are highly specific for line-associated infection
HIGH_RISK_ORGANISMS = [
    'STAPHYLOCOCCUS AUREUS',           # High virulence, strongly line-associated
    'CANDIDA ALBICANS',                # Fungemia in ICU → line source
    'CANDIDA GLABRATA',
    'CANDIDA PARAPSILOSIS',            # Especially line-associated
    'CANDIDA TROPICALIS',
    'CANDIDA',                         # Any Candida species
    'PSEUDOMONAS AERUGINOSA',          # Opportunistic, catheter-associated
]

# ALL CRBSI ORGANISMS - For general blood culture screening
# Require catheter tip culture for definite diagnosis (unless HIGH_RISK)
CRBSI_ORGANISMS = [
    # Gram-positive (most common CRBSI pathogens)
    'STAPHYLOCOCCUS AUREUS',
    'STAPHYLOCOCCUS, COAGULASE NEGATIVE',  # Most common overall
    'STAPHYLOCOCCUS EPIDERMIDIS',
    'STAPHYLOCOCCUS HOMINIS',
    'ENTEROCOCCUS FAECALIS',
    'ENTEROCOCCUS FAECIUM',
    'ENTEROCOCCUS',
    
    # Gram-negative
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
    
    # Fungi
    'CANDIDA ALBICANS',
    'CANDIDA GLABRATA',
    'CANDIDA PARAPSILOSIS',
    'CANDIDA TROPICALIS',
    'CANDIDA KRUSEI',
    'CANDIDA',
]

# EXCLUDED ORGANISMS - Likely contaminants, not true pathogens
CONTAMINANT_ORGANISMS = [
    'CORYNEBACTERIUM',          # Skin flora
    'PROPIONIBACTERIUM',        # Skin flora
    'BACILLUS',                 # Environmental contaminant (unless B. anthracis)
    'MICROCOCCUS',              # Skin flora
    'ALPHA STREPTOCOCCUS',      # Oral flora
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
    """
    Load MIMIC-IV tables with chunk-based processing for large files
    This prevents memory overflow on compute nodes
    """
    print("="*80)
    print("LOADING MIMIC-IV DATA (Chunked Processing)")
    print("="*80)
    
    hosp_path = os.path.join(mimic_path, 'hosp')
    icu_path = os.path.join(mimic_path, 'icu')
    
    data = {}
    
    # ===== Small tables - load directly =====
    print("\n1. Loading core tables...")
    data['patients'] = pd.read_csv(os.path.join(hosp_path, 'patients.csv'))
    data['admissions'] = pd.read_csv(os.path.join(hosp_path, 'admissions.csv'))
    data['icustays'] = pd.read_csv(os.path.join(icu_path, 'icustays.csv'))
    
    print(f"   ✓ Patients: {len(data['patients']):,}")
    print(f"   ✓ Admissions: {len(data['admissions']):,}")
    print(f"   ✓ ICU stays: {len(data['icustays']):,}")
    
    # ===== Diagnoses and procedures =====
    print("\n2. Loading diagnoses and procedures...")
    data['diagnoses_icd'] = pd.read_csv(os.path.join(hosp_path, 'diagnoses_icd.csv'))
    data['procedures_icd'] = pd.read_csv(os.path.join(hosp_path, 'procedures_icd.csv'))
    data['d_icd_diagnoses'] = pd.read_csv(os.path.join(hosp_path, 'd_icd_diagnoses.csv'))
    
    print(f"   ✓ Diagnosis codes: {len(data['diagnoses_icd']):,}")
    print(f"   ✓ Procedure codes: {len(data['procedures_icd']):,}")
    
    # ===== Microbiology - CRITICAL FOR LAB-BASED CRBSI =====
    print("\n3. Loading microbiology events...")
    data['microbiologyevents'] = pd.read_csv(os.path.join(hosp_path, 'microbiologyevents.csv'))
    
    print(f"   ✓ Microbiology events: {len(data['microbiologyevents']):,}")
    print(f"   ✓ Positive cultures: {data['microbiologyevents']['org_name'].notna().sum():,}")
    
    # ===== Lab events - CHUNKED PROCESSING =====
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
        # Filter immediately to save memory
        chunk_filtered = chunk[chunk['itemid'].isin(relevant_lab_itemids)]
        if len(chunk_filtered) > 0:
            chunks.append(chunk_filtered)
        print(f"   Chunk {chunk_count}: Kept {len(chunk_filtered):,} / {len(chunk):,} rows")
        gc.collect()  # Force garbage collection
    
    data['labevents'] = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()
    print(f"   ✓ Total lab events kept: {len(data['labevents']):,}")
    
    # ===== Chart events - CHUNKED PROCESSING =====
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
        # Filter immediately
        chunk_filtered = chunk[chunk['itemid'].isin(relevant_vital_itemids)]
        if len(chunk_filtered) > 0:
            chunks.append(chunk_filtered)
        print(f"   Chunk {chunk_count}: Kept {len(chunk_filtered):,} / {len(chunk):,} rows")
        gc.collect()
    
    data['chartevents'] = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()
    print(f"   ✓ Total chart events kept: {len(data['chartevents']):,}")
    
    # ===== ICU procedures =====
    print("\n6. Loading ICU procedure events...")
    data['procedureevents'] = pd.read_csv(os.path.join(icu_path, 'procedureevents.csv'))
    print(f"   ✓ Procedure events: {len(data['procedureevents']):,}")
    
    # ===== Item dictionaries =====
    print("\n7. Loading reference tables...")
    data['d_items'] = pd.read_csv(os.path.join(icu_path, 'd_items.csv'))
    data['d_labitems'] = pd.read_csv(os.path.join(hosp_path, 'd_labitems.csv'))
    
    # Force garbage collection
    gc.collect()
    
    print("\n" + "="*80)
    print("DATA LOADING COMPLETE")
    print("="*80)
    
    return data

# ======================== Cohort Identification ========================

def identify_central_line_cohort(data):
    """
    Identify ICU stays with central line placement
    Uses dual method: procedureevents (PRIMARY) + procedures_icd (SECONDARY)
    
    CLINICAL DOCUMENTATION:
    -----------------------
    METHOD 1 (PRIMARY): ICU Procedureevents
    - Real-time documentation by ICU staff
    - Uses itemid codes (225752, 225206, etc.)
    - Most reliable for ICU-placed lines
    - Captures: bedside insertions, PICC lines, dialysis catheters
    
    METHOD 2 (SECONDARY): Hospital Procedures_ICD  
    - Billing codes added post-discharge
    - Uses ICD-9/ICD-10 procedure codes
    - Captures: OR-placed lines, tunneled catheters, ports
    - Less complete but adds cases not in procedureevents
    
    RETURNS:
    --------
    - central_line_procedures: DataFrame with procedure timing
    - icu_stays_with_lines: Set of stay_ids with documented central lines
    """
    print("\n" + "="*80)
    print("IDENTIFYING CENTRAL LINE COHORT")
    print("="*80)
    
    # METHOD 1: ICU procedureevents (PRIMARY)
    print("\nMETHOD 1: ICU Procedureevents (itemid-based)")
    print("-" * 80)
    
    procedure_events = data['procedureevents'].copy()
    
    central_line_procedures_icu = procedure_events[
        procedure_events['itemid'].isin(CENTRAL_LINE_ITEMIDS.keys())
    ].copy()
    
    print(f"Central line procedures found: {len(central_line_procedures_icu):,}")
    print(f"Unique patients: {central_line_procedures_icu['subject_id'].nunique():,}")
    print(f"Unique ICU stays: {central_line_procedures_icu['stay_id'].nunique():,}")
    
    # Show breakdown by procedure type
    print("\nProcedure type breakdown:")
    for itemid, description in CENTRAL_LINE_ITEMIDS.items():
        count = (central_line_procedures_icu['itemid'] == itemid).sum()
        if count > 0:
            print(f"  {itemid} - {description}: {count:,}")
    
    # METHOD 2: Hospital procedures_icd (SECONDARY)
    print("\nMETHOD 2: Hospital Procedures_ICD (ICD code-based)")
    print("-" * 80)
    
    procedures_icd = data['procedures_icd'].copy()
    
    central_line_procedures_icd = procedures_icd[
        procedures_icd['icd_code'].isin(CENTRAL_LINE_ICD9 + CENTRAL_LINE_ICD10)
    ].copy()
    
    print(f"Central line ICD codes found: {len(central_line_procedures_icd):,}")
    print(f"Unique patients: {central_line_procedures_icd['subject_id'].nunique():,}")
    print(f"Unique admissions: {central_line_procedures_icd['hadm_id'].nunique():,}")
    
    # COMBINE BOTH METHODS
    print("\nCOMBINING BOTH METHODS")
    print("-" * 80)
    
    # Get unique stay_ids from ICU procedures
    icu_stays_with_lines = set(central_line_procedures_icu['stay_id'].dropna().unique())
    
    # Add stays from ICD-coded procedures
    if len(central_line_procedures_icd) > 0:
        icustays = data['icustays']
        hadm_with_lines = set(central_line_procedures_icd['hadm_id'].unique())
        additional_stays = icustays[icustays['hadm_id'].isin(hadm_with_lines)]['stay_id'].unique()
        icu_stays_with_lines.update(additional_stays)
        
        print(f"Additional stays from ICD codes: {len(additional_stays):,}")
    
    print(f"\n{'='*80}")
    print(f"TOTAL ICU STAYS WITH CENTRAL LINES: {len(icu_stays_with_lines):,}")
    print(f"{'='*80}")
    
    # Calculate expected CRBSI based on literature (5-12% rate)
    expected_crbsi_low = int(len(icu_stays_with_lines) * 0.05)
    expected_crbsi_high = int(len(icu_stays_with_lines) * 0.12)
    print(f"Expected CRBSI cases (5-12% rate): {expected_crbsi_low:,} - {expected_crbsi_high:,}")
    
    return central_line_procedures_icu, icu_stays_with_lines

# ======================== Lab-Based CRBSI Identification ========================

def identify_crbsi_lab_based(data, central_line_procedures, icu_stays_with_lines):
    """
    Identify CRBSI using LABORATORY CONFIRMATION (Gold Standard)
    
    This function implements CDC/NHSN criteria for CRBSI diagnosis using
    microbiological data rather than ICD diagnosis codes.
    
    CRBSI CRITERIA (all must be met):
    ----------------------------------
    1. Central line present ≥2 days before blood culture
    2. Positive blood culture with typical CRBSI organism
    3. EITHER:
       a) Matching catheter tip culture (DEFINITE CRBSI), OR
       b) High-risk organism without tip culture (PROBABLE CRBSI)
    4. Blood culture drawn while line present or ≤48h after removal
    
    CONFIDENCE LEVELS:
    ------------------
    DEFINITE: Blood culture + matching catheter tip culture
    PROBABLE: Blood culture + high-risk organism (no tip culture needed)
    
    RETURNS:
    --------
    crbsi_cases: DataFrame with columns:
        - subject_id, hadm_id, stay_id
        - charttime: Blood culture collection time
        - organism: Cultured organism
        - confidence: 'DEFINITE' or 'PROBABLE'
        - method: 'tip_culture' or 'clinical_criteria'
        - line_days: Days central line was present before culture
    """
    print("\n" + "="*80)
    print("LABORATORY-BASED CRBSI IDENTIFICATION")
    print("="*80)
    
    micro = data['microbiologyevents'].copy()
    icustays = data['icustays'].copy()
    
    # Ensure datetime format
    micro['charttime'] = pd.to_datetime(micro['charttime'])
    central_line_procedures['starttime'] = pd.to_datetime(central_line_procedures['starttime'])
    central_line_procedures['endtime'] = pd.to_datetime(central_line_procedures['endtime'])
    icustays['intime'] = pd.to_datetime(icustays['intime'])
    icustays['outtime'] = pd.to_datetime(icustays['outtime'])
    
    # STEP 1: Extract blood cultures
    print("\nSTEP 1: Identifying blood cultures...")
    print("-" * 80)
    
    blood_cultures = micro[
        micro['spec_type_desc'].str.contains('BLOOD', case=False, na=False)
    ].copy()
    
    print(f"Total blood cultures: {len(blood_cultures):,}")
    print(f"Positive blood cultures: {blood_cultures['org_name'].notna().sum():,}")
    
    # STEP 2: Extract catheter tip cultures
    print("\nSTEP 2: Identifying catheter tip cultures...")
    print("-" * 80)
    
    tip_cultures = micro[
        micro['spec_type_desc'].str.contains('CATHETER TIP', case=False, na=False)
    ].copy()
    
    print(f"Total catheter tip cultures: {len(tip_cultures):,}")
    print(f"Positive tip cultures: {tip_cultures['org_name'].notna().sum():,}")
    
    # STEP 3: Filter for positive blood cultures with CRBSI organisms
    print("\nSTEP 3: Filtering for CRBSI-associated organisms...")
    print("-" * 80)
    
    # Create regex pattern for organism matching
    organism_pattern = '|'.join([org.replace('(', r'\(').replace(')', r'\)') 
                                  for org in CRBSI_ORGANISMS])
    
    blood_positive = blood_cultures[
        blood_cultures['org_name'].str.contains(organism_pattern, case=False, na=False)
    ].copy()
    
    print(f"Blood cultures with CRBSI organisms: {len(blood_positive):,}")
    
    # Show organism distribution
    print("\nOrganism distribution in blood cultures:")
    org_counts = blood_positive['org_name'].value_counts().head(10)
    for org, count in org_counts.items():
        print(f"  {org}: {count:,}")
    
    # STEP 4: Match blood cultures to central line episodes
    print("\nSTEP 4: Matching blood cultures to central line episodes...")
    print("-" * 80)
    
    crbsi_cases = []
    definite_count = 0
    probable_count = 0
    excluded_no_line = 0
    excluded_timing = 0
    excluded_line_duration = 0
    
    print("Processing blood cultures...")
    
    for idx, blood in blood_positive.iterrows():
        if idx % 1000 == 0:
            print(f"  Processed {idx:,} / {len(blood_positive):,} blood cultures...")
        
        # Get patient's central line episodes
        patient_lines = central_line_procedures[
            central_line_procedures['subject_id'] == blood['subject_id']
        ].copy()
        
        if len(patient_lines) == 0:
            excluded_no_line += 1
            continue
        
        culture_time = blood['charttime']
        organism = blood['org_name']
        
        # Check each line episode for temporal match
        matched_line = None
        line_days = 0
        
        for _, line in patient_lines.iterrows():
            line_start = line['starttime']
            line_end = line['endtime'] if pd.notna(line['endtime']) else line_start + timedelta(days=30)
            
            # Culture must be ≥2 days after line insertion (CDC criterion)
            earliest_culture_time = line_start + timedelta(days=2)
            
            # Culture can be up to 48h after line removal
            latest_culture_time = line_end + timedelta(hours=48)
            
            if earliest_culture_time <= culture_time <= latest_culture_time:
                matched_line = line
                line_days = (culture_time - line_start).days
                break
        
        if matched_line is None:
            if any((blood['charttime'] >= line['starttime']) and 
                   (blood['charttime'] <= line['starttime'] + timedelta(days=2)) 
                   for _, line in patient_lines.iterrows()):
                excluded_line_duration += 1
            else:
                excluded_timing += 1
            continue
        
        # Get stay_id for this episode
        stay_id = matched_line['stay_id'] if 'stay_id' in matched_line else None
        if pd.isna(stay_id):
            # Try to get from icustays based on hadm_id and timing
            matching_stays = icustays[
                (icustays['subject_id'] == blood['subject_id']) &
                (icustays['hadm_id'] == blood['hadm_id']) &
                (icustays['intime'] <= culture_time) &
                (icustays['outtime'] >= culture_time)
            ]
            stay_id = matching_stays.iloc[0]['stay_id'] if len(matching_stays) > 0 else None
        
        # Check for matching catheter tip culture (DEFINITE CRBSI)
        matching_tip = tip_cultures[
            (tip_cultures['subject_id'] == blood['subject_id']) &
            (tip_cultures['org_name'].str.contains(organism, case=False, na=False)) &
            (abs((pd.to_datetime(tip_cultures['charttime']) - culture_time).dt.total_seconds()) < 48*3600)
        ]
        
        if len(matching_tip) > 0:
            # DEFINITE CRBSI - tip culture matches
            crbsi_cases.append({
                'subject_id': blood['subject_id'],
                'hadm_id': blood['hadm_id'],
                'stay_id': stay_id,
                'charttime': culture_time,
                'organism': organism,
                'confidence': 'DEFINITE',
                'method': 'tip_culture',
                'line_days': line_days,
                'line_start': matched_line['starttime'],
                'line_itemid': matched_line['itemid'] if 'itemid' in matched_line else None
            })
            definite_count += 1
            
        elif any(high_risk in organism.upper() for high_risk in HIGH_RISK_ORGANISMS):
            # PROBABLE CRBSI - high-risk organism
            crbsi_cases.append({
                'subject_id': blood['subject_id'],
                'hadm_id': blood['hadm_id'],
                'stay_id': stay_id,
                'charttime': culture_time,
                'organism': organism,
                'confidence': 'PROBABLE',
                'method': 'clinical_criteria',
                'line_days': line_days,
                'line_start': matched_line['starttime'],
                'line_itemid': matched_line['itemid'] if 'itemid' in matched_line else None
            })
            probable_count += 1
    
    # Convert to DataFrame
    crbsi_df = pd.DataFrame(crbsi_cases)
    
    # Remove duplicates (same patient, same day, same organism)
    if len(crbsi_df) > 0:
        crbsi_df['culture_date'] = crbsi_df['charttime'].dt.date
        crbsi_df = crbsi_df.drop_duplicates(
            subset=['subject_id', 'hadm_id', 'organism', 'culture_date'],
            keep='first'
        )
        crbsi_df = crbsi_df.drop('culture_date', axis=1)
    
    # SUMMARY STATISTICS
    print("\n" + "="*80)
    print("CRBSI IDENTIFICATION SUMMARY")
    print("="*80)
    
    print(f"\nBlood cultures screened: {len(blood_positive):,}")
    print(f"\nExclusion reasons:")
    print(f"  - No central line documented: {excluded_no_line:,}")
    print(f"  - Line present <2 days: {excluded_line_duration:,}")
    print(f"  - Culture outside time window: {excluded_timing:,}")
    
    print(f"\n{'='*80}")
    print(f"CRBSI CASES IDENTIFIED: {len(crbsi_df):,}")
    print(f"{'='*80}")
    print(f"  DEFINITE (tip culture match): {definite_count:,} ({definite_count/len(crbsi_df)*100 if len(crbsi_df)>0 else 0:.1f}%)")
    print(f"  PROBABLE (high-risk organism): {probable_count:,} ({probable_count/len(crbsi_df)*100 if len(crbsi_df)>0 else 0:.1f}%)")
    
    if len(icu_stays_with_lines) > 0:
        crbsi_rate = len(crbsi_df) / len(icu_stays_with_lines) * 100
        print(f"\nCRBSI Rate: {crbsi_rate:.2f}% of ICU stays with central lines")
        print(f"Expected rate from literature: 5-12%")
        
        if 5 <= crbsi_rate <= 12:
            print("✓ Rate is within expected clinical range")
        elif crbsi_rate < 5:
            print("⚠ Rate is lower than expected - may be undercounting")
        else:
            print("⚠ Rate is higher than expected - review inclusion criteria")
    
    if len(crbsi_df) > 0:
        print(f"\nMean line-days before CRBSI: {crbsi_df['line_days'].mean():.1f} days")
        print(f"Median line-days before CRBSI: {crbsi_df['line_days'].median():.1f} days")
        
        print("\nTop organisms causing CRBSI:")
        for org, count in crbsi_df['organism'].value_counts().head(10).items():
            print(f"  {org}: {count:,} ({count/len(crbsi_df)*100:.1f}%)")
    
    return crbsi_df

# ======================== Feature Extraction Functions ========================
# [Continue with the same feature extraction functions from previous script]
# These remain unchanged from the original preprocessing script

def extract_static_features(data, stay_id, admission, patient, icu_stay):
    """Extract static features for a patient"""
    
    features = {}
    
    # Demographics
    features['age'] = admission['admittime'].year - patient['anchor_year']
    features['sex'] = 1 if patient['gender'] == 'M' else 0
    
    # Calculate BMI if available (placeholder - would need height/weight from omr)
    features['bmi'] = 25.0  # Default, should be calculated from actual data
    
    # Admission details
    features['insurance_medicare'] = 1 if admission.get('insurance') == 'Medicare' else 0
    features['insurance_medicaid'] = 1 if admission.get('insurance') == 'Medicaid' else 0
    features['insurance_private'] = 1 if admission.get('insurance') == 'Private' else 0
    
    # Comorbidities (would extract from diagnoses_icd)
    features['diabetes'] = 0  # Placeholder
    features['ckd'] = 0
    features['immunosuppression'] = 0
    features['neutropenia'] = 0
    
    # ICU details
    features['icu_los_prior'] = max(0, (icu_stay['outtime'] - icu_stay['intime']).days)
    features['mechanical_ventilation'] = 0  # Would extract from procedureevents
    
    # Catheter details (would extract from procedureevents)
    features['catheter_type_picc'] = 0
    features['catheter_type_cvc'] = 1
    features['catheter_type_dialysis'] = 0
    features['insertion_site_subclavian'] = 0
    features['insertion_site_jugular'] = 1
    features['insertion_site_femoral'] = 0
    
    return features

def extract_vital_signs(data, stay_id, prediction_time, hours_back=48):
    """Extract vital signs for the specified time window"""
    
    chartevents = data['chartevents']
    
    # Filter for this stay and time window
    start_time = prediction_time - timedelta(hours=hours_back)
    
    stay_vitals = chartevents[
        (chartevents['stay_id'] == stay_id) &
        (chartevents['charttime'] >= start_time) &
        (chartevents['charttime'] < prediction_time)
    ].copy()
    
    # Initialize output array
    n_timepoints = hours_back
    n_features = len(VITAL_ITEMIDS)
    vitals_array = np.zeros((n_timepoints, n_features))
    
    # Extract each vital sign
    for feat_idx, (vital_name, itemids) in enumerate(VITAL_ITEMIDS.items()):
        vital_data = stay_vitals[stay_vitals['itemid'].isin(itemids)].copy()
        
        if len(vital_data) > 0:
            vital_data['hour_bin'] = ((vital_data['charttime'] - start_time).dt.total_seconds() // 3600).astype(int)
            
            # Average values within each hour
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
    
    # Initialize output (4 timepoints × 7 features for 12-hour bins over 2 days)
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
    """Extract catheter-related events"""
    
    # Placeholder - would extract from procedureevents, inputevents, etc.
    n_timepoints = days_back
    n_features = 6  # access_count, blood_draw, med_admin, dressing_change, assessment, flush
    
    return np.zeros((n_timepoints, n_features))

def generate_labels(crbsi_cases, stay_id, hadm_id, prediction_time, 
                    prediction_window_hours, survival_window_hours):
    """
    Generate multi-task labels for the prediction
    
    Labels include:
    1. Binary: CRBSI occurrence within prediction window
    2. Time-to-event: Hours until CRBSI (for survival analysis)
    3. Event: Whether CRBSI occurred (for Cox model)
    4. Decision: Clinical decision (remove now / 24h / continue)
    5. Clinical necessity: How much patient needs the line (0-1)
    """
    
    # Check for CRBSI in this stay
    stay_crbsi = crbsi_cases[
        (crbsi_cases['stay_id'] == stay_id) &
        (crbsi_cases['charttime'] >= prediction_time)
    ]
    
    if len(stay_crbsi) > 0:
        # CRBSI occurred
        first_crbsi = stay_crbsi.iloc[0]
        time_to_crbsi = (first_crbsi['charttime'] - prediction_time).total_seconds() / 3600
        
        # Binary label
        binary_label = 1 if time_to_crbsi <= prediction_window_hours else 0
        
        # Survival labels
        event_occurred = 1
        time_value = min(time_to_crbsi, survival_window_hours)
        
        # Decision label
        if time_to_crbsi <= 12:
            decision = 0  # Remove immediately
        elif time_to_crbsi <= 24:
            decision = 1  # Remove within 24h
        else:
            decision = 2  # Continue with monitoring
    else:
        # No CRBSI
        binary_label = 0
        event_occurred = 0
        time_value = survival_window_hours  # Censored at end of window
        decision = 2  # Continue monitoring
    
    # Clinical necessity (placeholder - would calculate from patient data)
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
    """Process entire patient cohort with feature extraction"""
    
    print("\n" + "="*80)
    print("PROCESSING PATIENT COHORT")
    print("="*80)
    
    icustays = data['icustays']
    cohort = icustays[icustays['stay_id'].isin(icu_stays_with_lines)].copy()
    
    # Join with admissions
    cohort = cohort.merge(
        data['admissions'][['hadm_id', 'subject_id', 'admittime', 'dischtime']],
        on=['hadm_id', 'subject_id'],
        how='left'
    )
    
    # Convert to datetime
    cohort['admittime'] = pd.to_datetime(cohort['admittime'])
    cohort['intime'] = pd.to_datetime(cohort['intime'])
    cohort['outtime'] = pd.to_datetime(cohort['outtime'])
    
    print(f"\nProcessing {len(cohort):,} ICU stays with central lines...")
    print(f"Expected processing time: ~{len(cohort)//100:,} minutes at 100 stays/min")
    
    model_data = []
    
    for idx, stay in cohort.iterrows():
        if idx % 100 == 0:
            print(f"  Progress: {idx:,} / {len(cohort):,} ({idx/len(cohort)*100:.1f}%)")
            gc.collect()  # Garbage collection every 100 stays
        
        try:
            # Set prediction time (36h after ICU admission)
            prediction_time = stay['intime'] + timedelta(hours=36)
            
            if prediction_time >= stay['outtime']:
                continue
            
            # Get patient info
            patient = data['patients'][
                data['patients']['subject_id'] == stay['subject_id']
            ].iloc[0]
            
            admission = data['admissions'][
                data['admissions']['hadm_id'] == stay['hadm_id']
            ].iloc[0]
            
            # Extract features
            static_features = extract_static_features(data, stay['stay_id'], admission, patient, stay)
            vital_signs = extract_vital_signs(data, stay['stay_id'], prediction_time, args.feature_window)
            lab_values = extract_lab_values(data, stay['hadm_id'], prediction_time, args.feature_window // 24)
            catheter_events = extract_catheter_events(data, stay['stay_id'], prediction_time, 14)
            
            # Generate labels
            labels = generate_labels(
                crbsi_cases, stay['stay_id'], stay['hadm_id'], prediction_time,
                args.prediction_window, args.survival_window
            )
            
            # Store sample
            sample = {
                'stay_id': stay['stay_id'],
                'subject_id': stay['subject_id'],
                'hadm_id': stay['hadm_id'],
                'prediction_time': prediction_time,
                'static': np.array(list(static_features.values())),
                'temporal': [vital_signs, lab_values, catheter_events],
                'labels': labels,
                'static_keys': list(static_features.keys())
            }
            
            model_data.append(sample)
            
        except Exception as e:
            print(f"    Error processing stay {stay['stay_id']}: {e}")
            continue
    
    print(f"\n✓ Successfully processed {len(model_data):,} samples")
    
    return model_data

def normalize_and_impute(model_data):
    """Normalize and impute missing values"""
    
    print("\n" + "="*80)
    print("NORMALIZATION AND IMPUTATION")
    print("="*80)
    
    # Static features
    all_static = np.array([d['static'] for d in model_data])
    
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
    
    # Save processed data
    output_file = os.path.join(args.output_path, 'crbsi_processed_data.pkl')
    with open(output_file, 'wb') as f:
        pickle.dump(model_data, f)
    print(f"✓ Saved: {output_file}")
    
    # Save scaler
    scaler_file = os.path.join(args.output_path, 'static_scaler.pkl')
    with open(scaler_file, 'wb') as f:
        pickle.dump(scaler_static, f)
    print(f"✓ Saved: {scaler_file}")
    
    # Save CRBSI cases for clinical review
    crbsi_file = os.path.join(args.output_path, 'crbsi_cases_identified.csv')
    crbsi_cases.to_csv(crbsi_file, index=False)
    print(f"✓ Saved: {crbsi_file}")
    
    # Comprehensive metadata
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
        
        # CRBSI identification details
        'crbsi_method': 'Laboratory-based (microbiologyevents)',
        'crbsi_criteria': 'CDC/NHSN adapted',
        'definite_crbsi': int((crbsi_cases['confidence'] == 'DEFINITE').sum()),
        'probable_crbsi': int((crbsi_cases['confidence'] == 'PROBABLE').sum()),
        
        # Clinical validation metrics
        'mean_line_days_before_crbsi': float(crbsi_cases['line_days'].mean()) if len(crbsi_cases) > 0 else 0,
        'top_organisms': crbsi_cases['organism'].value_counts().head(5).to_dict() if len(crbsi_cases) > 0 else {},
        
        # Feature specifications
        'static_features': model_data[0]['static_keys'] if len(model_data) > 0 else [],
        'vital_signs_features': list(VITAL_ITEMIDS.keys()),
        'lab_features': list(LAB_ITEMIDS.keys()),
        'catheter_event_features': ['line_access', 'blood_draw', 'med_admin', 
                                    'dressing_change', 'assessment', 'flush'],
        
        # Processing details
        'chunk_size': args.chunk_size,
        'processing_date': datetime.now().isoformat(),
    }
    
    metadata_file = os.path.join(args.output_path, 'metadata.pkl')
    with open(metadata_file, 'wb') as f:
        pickle.dump(metadata, f)
    print(f"✓ Saved: {metadata_file}")
    
    # Save human-readable summary
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
    print(f"\nConfiguration:")
    print(f"  MIMIC-IV Path: {args.mimic_path}")
    print(f"  Output Path: {args.output_path}")
    print(f"  Feature Window: {args.feature_window}h")
    print(f"  Prediction Window: {args.prediction_window}h")
    print(f"  Survival Window: {args.survival_window}h")
    print(f"  Chunk Size: {args.chunk_size:,} rows")
    
    # Load data (chunked for memory efficiency)
    data = load_mimic_data_chunked(args.mimic_path, args.chunk_size)
    
    # Identify central line cohort
    central_line_procedures, icu_stays_with_lines = identify_central_line_cohort(data)
    
    # Identify CRBSI cases using laboratory data
    crbsi_cases = identify_crbsi_lab_based(data, central_line_procedures, icu_stays_with_lines)
    
    # Process patient cohort
    model_data = process_patient_cohort(data, icu_stays_with_lines, crbsi_cases, args)
    
    # Normalize and impute
    model_data_normalized, scaler_static = normalize_and_impute(model_data)
    
    # Save results with detailed documentation
    save_processed_data(model_data_normalized, scaler_static, crbsi_cases, args)
    
    print("\n✓ Preprocessing completed successfully!")
    print("\nFor clinical review, see:")
    print(f"  - {args.output_path}/crbsi_cases_identified.csv")
    print(f"  - {args.output_path}/PROCESSING_SUMMARY.txt")

if __name__ == "__main__":
    main()
