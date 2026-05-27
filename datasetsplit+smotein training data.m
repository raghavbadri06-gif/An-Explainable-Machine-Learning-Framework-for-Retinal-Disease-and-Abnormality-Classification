clc; clear; close all;

%% =========================================================
% COMPLETE DATA PREPARATION WITH SMOTE (KNN VERSION)
% Split: 80% Train, 10% Validation, 10% Test
% SMOTE applied ONLY to training set
% Target: 110 samples per class in training set
% =========================================================

%% STEP 1: Load Dataset
[file, path] = uigetfile({'*.csv;*.xlsx','Data Files (*.csv, *.xlsx)'});
if isequal(file,0)
    error('No file selected');
end

data = readtable(fullfile(path,file));

% Assume LAST column = label
X = data{:,1:end-1};
Y = categorical(data{:,end});

featureNames = data.Properties.VariableNames(1:end-1);
labelName    = data.Properties.VariableNames{end};

fprintf('\n========================================\n');
fprintf('DATASET LOADED\n');
fprintf('========================================\n');
fprintf('File: %s\n', file);
fprintf('Samples: %d\n', size(X,1));
fprintf('Features: %d\n', size(X,2));
fprintf('Classes: %s\n', strjoin(cellstr(categories(Y)), ', '));

%% STEP 2: DATA CLEANING
fprintf('\n========================================\n');
fprintf('CLEANING DATA\n');
fprintf('========================================\n');

% Replace Inf with NaN
X(~isfinite(X)) = NaN;

% Fill NaN using column median (manual)
for i = 1:size(X,2)
    col = X(:,i);
    col_clean = col(~isnan(col));
    if ~isempty(col_clean)
        med = median(col_clean);
    else
        med = 0;
    end
    
    if isnan(med) % if entire column is NaN
        med = 0;
    end
    
    col(isnan(col)) = med;
    X(:,i) = col;
end

% Remove outliers using IQR
for i = 1:size(X,2)
    Q1 = quantile(X(:,i),0.25);
    Q3 = quantile(X(:,i),0.75);
    IQR = Q3 - Q1;

    lower = Q1 - 1.5 * IQR;
    upper = Q3 + 1.5 * IQR;

    idx = (X(:,i) < lower) | (X(:,i) > upper);
    X(idx,i) = median(X(:,i));
end

% Remove duplicate rows
[X, ia, ~] = unique(X,'rows');
Y = Y(ia);

fprintf('After cleaning: %d samples\n', size(X,1));

%% STEP 3: NORMALIZATION
fprintf('\n========================================\n');
fprintf('NORMALIZING FEATURES\n');
fprintf('========================================\n');

% Z-score normalization
X = normalize(X, 'zscore');
fprintf('Features normalized (z-score)\n');

%% STEP 4: STRATIFIED SPLIT (80% Train, 10% Validation, 10% Test)
fprintf('\n========================================\n');
fprintf('SPLITTING DATA\n');
fprintf('========================================\n');

rng(42); % For reproducibility

% Convert to categorical for splitting
Y_cat = Y;
n = size(X,1);

% Initialize indices
train_idx = [];
val_idx = [];
test_idx = [];

% Split each class proportionally
classes = categories(Y_cat);
for i = 1:length(classes)
    class_indices = find(Y_cat == classes{i});
    n_class = length(class_indices);
    
    % Shuffle
    class_indices = class_indices(randperm(n_class));
    
    % Calculate splits
    n_train = round(n_class * 0.8);
    n_val = round(n_class * 0.1);
    
    % Assign (ensure at least 1 sample for val and test if possible)
    if n_train < 1
        n_train = 1;
    end
    if n_val < 1 && n_class > 2
        n_val = 1;
    end
    
    % Assign indices
    train_idx = [train_idx; class_indices(1:n_train)];
    if n_val > 0 && (n_train + n_val) <= n_class
        val_idx = [val_idx; class_indices(n_train+1:n_train+n_val)];
        test_idx = [test_idx; class_indices(n_train+n_val+1:end)];
    else
        test_idx = [test_idx; class_indices(n_train+1:end)];
    end
end

% Shuffle each set
train_idx = train_idx(randperm(length(train_idx)));
if ~isempty(val_idx)
    val_idx = val_idx(randperm(length(val_idx)));
end
test_idx = test_idx(randperm(length(test_idx)));

% Create datasets
X_train = X(train_idx, :);
Y_train = Y(train_idx);
X_val = X(val_idx, :);
Y_val = Y(val_idx);
X_test = X(test_idx, :);
Y_test = Y(test_idx);

fprintf('Training set: %d samples (%.1f%%)\n', length(Y_train), length(Y_train)/n*100);
fprintf('Validation set: %d samples (%.1f%%)\n', length(Y_val), length(Y_val)/n*100);
fprintf('Test set: %d samples (%.1f%%)\n', length(Y_test), length(Y_test)/n*100);

%% STEP 5: SHOW ORIGINAL TRAINING CLASS DISTRIBUTION
fprintf('\n========================================\n');
fprintf('ORIGINAL TRAINING CLASS DISTRIBUTION\n');
fprintf('========================================\n');

% Manual tabulate (without Statistics Toolbox)
[unique_classes, ~, class_idx] = unique(Y_train);
class_counts = histcounts(class_idx, length(unique_classes));
for i = 1:length(unique_classes)
    fprintf('  %s: %d samples\n', char(unique_classes(i)), class_counts(i));
end

%% STEP 6: APPLY SMOTE TO TRAINING SET ONLY (TARGET: 110 samples per class)
fprintf('\n========================================\n');
fprintf('APPLYING SMOTE (KNN VERSION) TO TRAINING SET\n');
fprintf('Target: 110 samples per class\n');
fprintf('========================================\n');

% Set target samples per class to 110
target_samples_per_class = 110;
k = 5; % Number of nearest neighbors for SMOTE

try
    [X_train_smote, Y_train_smote] = smote_knn_with_target(X_train, Y_train, k, target_samples_per_class);
    fprintf('\n✓ SMOTE completed successfully!\n');
    fprintf('Training set after SMOTE: %d samples\n', length(Y_train_smote));
    fprintf('Target achieved: %d samples per class\n', target_samples_per_class);
catch ME
    fprintf('\n✗ SMOTE error: %s\n', ME.message);
    fprintf('Using original training set without SMOTE...\n');
    X_train_smote = X_train;
    Y_train_smote = Y_train;
end

%% STEP 7: SHOW BALANCED CLASS DISTRIBUTION
fprintf('\n========================================\n');
fprintf('BALANCED TRAINING CLASS DISTRIBUTION (After SMOTE)\n');
fprintf('========================================\n');

% Manual tabulate for balanced data
[unique_classes_bal, ~, class_idx_bal] = unique(Y_train_smote);
class_counts_bal = histcounts(class_idx_bal, length(unique_classes_bal));
for i = 1:length(unique_classes_bal)
    fprintf('  %s: %d samples\n', char(unique_classes_bal(i)), class_counts_bal(i));
end

%% STEP 8: CREATE TABLES FOR SAVING
fprintf('\n========================================\n');
fprintf('SAVING DATASETS\n');
fprintf('========================================\n');

% Create balanced training table
train_balanced = array2table(X_train_smote, 'VariableNames', featureNames);
train_balanced.(labelName) = Y_train_smote;

% Create validation table (if exists)
if ~isempty(X_val)
    val_table = array2table(X_val, 'VariableNames', featureNames);
    val_table.(labelName) = Y_val;
else
    val_table = [];
end

% Create test table
test_table = array2table(X_test, 'VariableNames', featureNames);
test_table.(labelName) = Y_test;

% Save files
train_file = fullfile(path, 'train_balanced_smote_110.csv');
val_file = fullfile(path, 'validation.csv');
test_file = fullfile(path, 'test.csv');

try
    writetable(train_balanced, train_file);
    fprintf('✓ Saved: %s\n', train_file);
    fprintf('  Size: %d rows × %d columns\n', height(train_balanced), width(train_balanced));
    fprintf('  Last column: %s (response variable)\n\n', labelName);
catch ME
    fprintf('✗ Error saving training file: %s\n', ME.message);
end

try
    if ~isempty(val_table)
        writetable(val_table, val_file);
        fprintf('✓ Saved: %s\n', val_file);
        fprintf('  Size: %d rows × %d columns\n\n', height(val_table), width(val_table));
    else
        fprintf('⚠ No validation data to save\n\n');
    end
catch ME
    fprintf('✗ Error saving validation file: %s\n', ME.message);
end

try
    writetable(test_table, test_file);
    fprintf('✓ Saved: %s\n', test_file);
    fprintf('  Size: %d rows × %d columns\n', height(test_table), width(test_table));
catch ME
    fprintf('✗ Error saving test file: %s\n', ME.message);
end

%% STEP 9: DISPLAY FINAL INFORMATION
fprintf('\n========================================\n');
fprintf('FILES READY FOR CLASSIFICATION LEARNER\n');
fprintf('========================================\n');

fprintf('\nAll files saved in:\n');
fprintf('  %s\n\n', path);

fprintf('1. TRAINING (BALANCED with KNN-SMOTE to %d samples/class):\n', target_samples_per_class);
fprintf('   %s\n', train_file);
fprintf('   Training samples: %d\n', height(train_balanced));
fprintf('   Each class: %d samples\n\n', target_samples_per_class);

fprintf('2. VALIDATION:\n');
fprintf('   %s\n', val_file);
if ~isempty(val_table)
    fprintf('   Validation samples: %d\n\n', height(val_table));
else
    fprintf('   No validation data\n\n');
end

fprintf('3. TEST:\n');
fprintf('   %s\n', test_file);
fprintf('   Test samples: %d\n', height(test_table));

%% STEP 10: INSTRUCTIONS FOR CLASSIFICATION LEARNER
fprintf('\n========================================\n');
fprintf('INSTRUCTIONS FOR CLASSIFICATION LEARNER\n');
fprintf('========================================\n');

fprintf('\nSTEP 1: Open Classification Learner App\n');
fprintf('   >> classificationLearner\n\n');

fprintf('STEP 2: Import Training Data\n');
fprintf('   - Click "New Session" → "From File"\n');
fprintf('   - Navigate to: %s\n', path);
fprintf('   - Select: train_balanced_smote_110.csv\n');
fprintf('   - Set response variable to: %s (last column)\n', labelName);
fprintf('   - Click "Start Session"\n\n');

fprintf('STEP 3: Train Models\n');
fprintf('   - Select models: SVM, Tree, Ensemble, etc.\n');
fprintf('   - Click "Train" or "Train All"\n');
fprintf('   - Use 5-fold cross-validation\n\n');

fprintf('STEP 4: Test Your Best Model\n');
fprintf('   - Select your best trained model\n');
fprintf('   - Click "Test" → "Test with File"\n');
fprintf('   - Select: test.csv\n');
fprintf('   - Review the confusion matrix and accuracy\n\n');

fprintf('STEP 5: Export Model\n');
fprintf('   - Click "Export" → "Export Model"\n');
fprintf('   - Save for future predictions\n\n');

fprintf('========================================\n');
fprintf('✅ COMPLETE! Data is ready for Classification Learner\n');
fprintf('========================================\n');

%% =========================================================
% FUNCTION: KNN-BASED SMOTE WITH TARGET SAMPLES PER CLASS
% =========================================================
function [Xsm, Ysm] = smote_knn_with_target(X, Y, k, targetCount)

classes = categories(Y);
Xsm = [];
Ysm = [];

for i = 1:numel(classes)

    classLabel = classes{i};
    idx = (Y == classLabel);
    Xi = X(idx,:);
    ni = size(Xi,1);

    % Add original samples
    Xsm = [Xsm; Xi];
    Ysm = [Ysm; categorical(repmat({classLabel}, ni,1))];

    if ni < targetCount

        nToGenerate = targetCount - ni;

        % Handle small class safely
        if ni > 1
            k_eff = min(k, ni-1);
        else
            k_eff = 1;
        end

        % Calculate distances
        if ni > 1
            distances = pdist2(Xi, Xi);
            [~, neighbors] = sort(distances, 2);
        end

        syntheticSamples = zeros(nToGenerate, size(X,2));

        for j = 1:nToGenerate

            randIdx = randi(ni);
            x_i = Xi(randIdx,:);

            if ni > 1
                nn = neighbors(randIdx, 2:k_eff+1);
                neighborIdx = nn(randi(length(nn)));
                x_nn = Xi(neighborIdx,:);
            else
                % Only 1 sample case - add small noise
                x_nn = x_i;
                noise = randn(1, size(X,2)) * 0.01;
                syntheticSamples(j,:) = x_i + noise;
                continue;
            end

            lambda = rand;
            syntheticSamples(j,:) = x_i + lambda*(x_nn - x_i);
        end

        Xsm = [Xsm; syntheticSamples];
        Ysm = [Ysm; categorical(repmat({classLabel}, nToGenerate,1))];
    elseif ni > targetCount
        % If class has more than target, randomly undersample
        fprintf('  Warning: %s has %d samples, reducing to %d\n', classLabel, ni, targetCount);
        keep_idx = randperm(ni, targetCount);
        Xsm = [Xsm; Xi(keep_idx, :)];
        Ysm = [Ysm; categorical(repmat({classLabel}, targetCount, 1))];
    end
end

% Shuffle dataset
perm = randperm(size(Xsm,1));
Xsm = Xsm(perm,:);
Ysm = Ysm(perm);

end
