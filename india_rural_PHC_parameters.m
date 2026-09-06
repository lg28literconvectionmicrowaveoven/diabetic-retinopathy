% ==============================================================================
% india_rural_PHC_parameters.m
% Rural India Primary Health Centre (PHC) Telemedicine Deployment Assumptions
% Operational, Demographic, and Telecommunications Profiles for Simulink/SimEvents
% Sources: National Health Mission (NHM), National Programme for Control of Blindness (NPCB)
% ==============================================================================

clear PHC_params;
PHC_params = struct();

% ------------------------------------------------------------------------------
% 1. FACILITY & NETWORK INFRASTRUCTURE (ENGINEERING ASSUMPTIONS & PUBLISHED STANDARDS)
% ------------------------------------------------------------------------------
PHC_params.number_of_PHCs                     = 10;            % Regional hub cluster of primary health centres
PHC_params.patients_per_PHC_per_day           = 40;            % Average daily diabetic and hypertension screening queue
PHC_params.operating_days_per_year            = 250;           % Working days per year (excluding public holidays & Sundays)
PHC_params.operating_hours_per_day            = 6.0;           % Daily clinical operating window (10:00 AM - 4:00 PM)
PHC_params.target_patients_per_year           = PHC_params.number_of_PHCs * PHC_params.patients_per_PHC_per_day * PHC_params.operating_days_per_year;

% ------------------------------------------------------------------------------
% 2. ACQUISITION HARDWARE & PHC WORKFLOW
% ------------------------------------------------------------------------------
PHC_params.cameras_per_PHC                    = 1;             % Portable non-mydriatic fundus camera (e.g. Remidio / Forus 3nethra)
PHC_params.camera_acquisition_time_sec        = 45.0;          % Operator positioning, focus, and retinal flash per patient (both eyes)
PHC_params.patient_registration_time_sec      = 60.0;          % Initial token generation, Aadhaar/ABHA ID registration
PHC_params.patient_dilation_rate              = 0.05;          % Patients requiring tropicamide pupil dilation due to small pupils
PHC_params.dilation_wait_time_sec             = 1200.0;        % 20-minute waiting time if dilation is mandated

% ------------------------------------------------------------------------------
% 3. TELEMEDICINE TELECOMMUNICATIONS & BANDWIDTH (RURAL INDIA CELLULAR/BHARATNET)
% ------------------------------------------------------------------------------
PHC_params.network_bandwidth_Mbps             = 4.0;           % Dedicated rural broadband / 4G cellular uplink bandwidth
PHC_params.network_latency_ms                 = 85.0;          % Round-trip network latency to district telemedicine cloud server
PHC_params.network_packet_drop_rate           = 0.015;         % Rural packet loss / retry probability

% ------------------------------------------------------------------------------
% 4. DISTRICT HOSPITAL & HUMAN OPHTHALMOLOGIST REVIEW WORKLOAD
% ------------------------------------------------------------------------------
PHC_params.number_of_remote_ophthalmologists  = 2;             % Dedicated specialists at District Referral Hospital
PHC_params.ophthalmologist_review_time_sec    = 90.0;          % Review time per flagged case (fundus photo + Grad-CAM heatmap inspection)
PHC_params.ophthalmologist_work_hours_per_day = 5.0;           % Dedicated daily tele-consultation time
PHC_params.teleconsultation_referral_threshold = 2;            % DR Grade >= 2 (Moderate NPDR or worse) referred to district tertiary center

% ------------------------------------------------------------------------------
% 5. EDGE INFERENCE SERVERS
% ------------------------------------------------------------------------------
PHC_params.number_of_AI_workers               = 1;             % Low-power on-site edge compute box per PHC (e.g. Jetson / Mac Mini / M4 Edge)
PHC_params.edge_power_consumption_watts       = 25.0;          % Power rating of edge inference unit
PHC_params.battery_backup_hours               = 4.0;           % UPS backup capacity during rural grid power shedding

fprintf('[SUCCESS] Loaded india_rural_PHC_parameters into workspace.\n');
