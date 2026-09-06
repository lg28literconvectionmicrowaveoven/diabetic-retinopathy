% ==============================================================================
% DR_simulink_parameters.m
% Complete Measured Parameters for Rural India PHC Diabetic Retinopathy Simulation
% System: MedSigLIP (SO400M, 1152-dim) + MLP Head + Vectorized Grad-CAM
% Hardware Measured: Apple M4 (10-core), Apple Silicon MPS GPU, Batch Size = 1
% Dataset Provenance: IDRiD (Indian Dataset, n=103) & Messidor-2 (n=1744)
% Generated: 2026-09-06 09:16:37
% ==============================================================================

clear DR_params;
DR_params = struct();

% ------------------------------------------------------------------------------
% 1. AI COMPUTATIONAL TIMING & THROUGHPUT (MODEL-MEASURED)
% ------------------------------------------------------------------------------
DR_params.ai_inference_time_sec               = 1.174922;    % Mean pure AI inference latency per image (sec)
DR_params.ai_inference_time_std_sec           = 0.253478;
DR_params.ai_inference_time_p95_sec           = 1.743991;

DR_params.preprocessing_time_sec              = 0.317230;   % Illumination + Green-channel CLAHE + Mask + Norm
DR_params.encoder_backbone_time_sec           = 0.761125;  % MedSigLIP ViT forward time
DR_params.pooling_time_sec                    = 0.000016;         % Attention pooling head time
DR_params.classifier_time_sec                 = 0.002480;      % MLP classification head time

DR_params.gradcam_generation_time_sec         = 0.410059; % Vectorized analytical VJP Grad-CAM (5 grades + overlay)
DR_params.gradcam_generation_time_std_sec     = 0.205007;
DR_params.gradcam_generation_time_p95_sec     = 0.807125;

DR_params.total_ai_latency_with_gradcam_sec   = 1.667248; % Full end-to-end turnaround
DR_params.throughput_images_per_sec          = 0.5998;
DR_params.throughput_images_per_hour         = 2159.25;

DR_params.model_total_parameters              = 444056853;     % Vision Transformer + MLP parameters
DR_params.peak_memory_mb                      = 1004.50;       % Resident memory footprint on edge device

% ------------------------------------------------------------------------------
% 2. IMAGE QUALITY & SCREENING GATE (DATASET-DERIVED & MODEL-MEASURED)
% ------------------------------------------------------------------------------
DR_params.image_quality_assessment_time_sec   = 0.082254;
DR_params.acceptable_image_rate               = 0.5146;    % Quality gate: acceptable fundus scans
DR_params.borderline_image_rate               = 0.4757;    % Usable / borderline scans
DR_params.ungradeable_image_rate              = 0.0097;   % Poor quality scans requiring re-capture
DR_params.enhancement_trigger_rate            = 0.4854; % Trigger rate for adaptive illumination/CLAHE

% ------------------------------------------------------------------------------
% 3. EPIDEMIOLOGICAL DR GRADE PREVALENCE (DATASET-DERIVED: IDRiD Indian Cohort)
% ------------------------------------------------------------------------------
DR_params.grade0_rate                         = 0.3301;    % No Diabetic Retinopathy
DR_params.grade1_rate                         = 0.0485;    % Mild Non-Proliferative DR
DR_params.grade2_rate                         = 0.3107;    % Moderate Non-Proliferative DR
DR_params.grade3_rate                         = 0.1845;    % Severe Non-Proliferative DR
DR_params.grade4_rate                         = 0.1262;    % Proliferative Diabetic Retinopathy

DR_params.referable_dr_prevalence             = 0.6214; % Grade >= 2
DR_params.non_referable_dr_prevalence         = 0.3786; % Grade 0-1

% ------------------------------------------------------------------------------
% 4. CLINICAL DIAGNOSTIC ACCURACY (MODEL-MEASURED on IDRiD)
% ------------------------------------------------------------------------------
DR_params.referable_DR_sensitivity            = 0.9688; % True Positive Rate (Recall)
DR_params.referable_DR_specificity            = 0.4872; % True Negative Rate (1 - FPR)
DR_params.referable_DR_ppv                    = 0.7561;         % Positive Predictive Value
DR_params.referable_DR_npv                    = 0.9048;         % Negative Predictive Value
DR_params.referable_DR_auc                    = 0.8962;         % Area Under ROC Curve
DR_params.multiclass_accuracy                 = 0.3204;
DR_params.multiclass_macro_f1                 = 0.2175;
DR_params.quadratic_weighted_kappa            = 0.4113;

% ------------------------------------------------------------------------------
% 5. PREDICTION CONFIDENCE & CALIBRATION (MODEL-MEASURED)
% ------------------------------------------------------------------------------
DR_params.average_prediction_confidence       = 0.7466;
DR_params.median_prediction_confidence        = 0.8395;
DR_params.expected_calibration_error          = 0.4269;  % ECE (10 bins)
DR_params.brier_score                         = 1.1163;

% ------------------------------------------------------------------------------
% 6. HUMAN REVIEW WORKLOAD IN RURAL PHC TRIAGE (MODEL-DERIVED)
% ------------------------------------------------------------------------------
DR_params.human_review_rate                   = 0.9515;   % % cases triaged to human ophthalmologist
DR_params.auto_cleared_rate                   = 0.0485;   % % cases safely cleared at PHC level
DR_params.explainability_time_per_review_sec  = 0.410059;

% ------------------------------------------------------------------------------
% 7. TELEMEDICINE IMAGE & NETWORK TRANSMISSION (DATASET-DERIVED)
% ------------------------------------------------------------------------------
DR_params.raw_image_size_mean_MB              = 0.4277;  % Uncompressed/raw fundus capture file size
DR_params.raw_image_size_p95_MB               = 0.7511;
DR_params.acquisition_image_width             = 4288;
DR_params.acquisition_image_height            = 2848;
DR_params.preprocessed_image_width            = 384;
DR_params.preprocessed_image_height           = 384;

fprintf('[SUCCESS] Loaded DR_simulink_parameters into workspace.\n');
