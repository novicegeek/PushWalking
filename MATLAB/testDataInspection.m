%% Set parameters and data paths
clc
clear
close all

cd 'D:\Overseas\German Sport University Cologne\20.Course Materials\TSM11-Project&Applied Research Methods\24.Scripts'
fpFreq = 2400;
padFreq = 5000;
cutOffFreq = 15;
r = 0.06375;
angles = [36, 60, 30];

% Recalibrated by Thomas F Januar 2022 (given in V/5000 N) 
scaleFactors = [0.69793, 0.5698, 0.5536, 0.53573, 0.49673, 0.48505, ...
                0.5954, 0.6032, 0.58867, 0.6513, 0.6032, 0.61773];
scaleFactors = 5000./scaleFactors;

%% Read, process and plot data
close all
subjectID = "S02";
date = "20260126";
subjectFolder = subjectID + "_" + date;
subjectDir = fullfile("..\22.Measurement Data\", subjectFolder);
fpDir = fullfile(subjectDir, "Qualisys");
padDir = fullfile(subjectDir, "Boxing Pad");
subPlotNRows = 4;
subPlotNCols = 6;

% t = tiledlayout(2, length(speeds), TileIndexing="columnmajor", TileSpacing="compact");
% syncPeakFrame = [21416,25438,29277;23703,23805,33791];  % The number of frame when the synchronization peak occurs

% Extract trial list
trialListFile = fullfile(subjectDir, subjectFolder + "_trial_list.csv");
opts = detectImportOptions(trialListFile, VariableNamingRule="preserve");
opts = setvaropts(opts, 'Invalid Trials', 'FillValue', 0);
trialList = readtable(trialListFile, opts);
trialList.("Recorded Trial No.") = zeros(height(trialList), 1);
for i = 1:height(trialList)
    if trialList(i, "Block Trial No.").Variables == 1
        trialList(i, "Recorded Trial No.").Variables = 1 + trialList(i, "Invalid Trials").Variables;
    else
        trialList(i, "Recorded Trial No.").Variables = trialList(i-1, "Recorded Trial No.").Variables + 1 + trialList(i, "Invalid Trials").Variables;
    end
end     

% Traverse measurement data
fpFiles = dir(fpDir);
fpFiles = {fpFiles.name};
padFiles = dir(padDir);
padFiles = {padFiles.name};
for file = horzcat(fpFiles, padFiles)
    [~, fileNoExt, ext] = fileparts(file{1});
    switch ext
        case ".mat"
            [speed, speedInd, trialInd] = getSpeedTrialInd(fileNoExt, "Qualisys");
        case ".txt"
            [speed, speedInd, trialInd] = getSpeedTrialInd(fileNoExt, "Boxing Pad");
        otherwise
            continue
    end
    
    % Skip the file if it is an invalid trial
    trialListInd = find(string(trialList.("Block Speed")) == lower(speed) & trialList.("Recorded Trial No.") == trialInd);
    if ~any(trialListInd)
        continue
    else
        trialInd = trialList(trialListInd, "Block Trial No.").Variables;
    end

    % Force plate data
    if strcmp(ext, ".mat")
        figure(speedInd)
        [b, a] = butter(2, cutOffFreq/(fpFreq/2), "low");
        data = load(fullfile(fpDir, file{1}));
        data = data.(fileNoExt).Force;
        subplot(subPlotNRows, subPlotNCols, trialInd)
        for i = 1:2
            dataFiltered = filter(b, a, transpose(data(i).Force));
            dataOffset = dataFiltered - mean(dataFiltered(0.1*fpFreq:0.3*fpFreq,:));
    
            nRows = height(dataOffset);
            plot((1:nRows)/fpFreq, dataOffset(:,3), 'b', LineWidth=1)
            title(sprintf("Trial No.%d  %s", trialInd, string(trialList(trialListInd, "Intensity").Variables)))
            hold on
        end
    % Boxing pad data
    else
        figure(speedInd + 3)
        [b, a] = butter(2, cutOffFreq/(padFreq/2), "low");
        data = readmatrix(fullfile(padDir, file{1}));
        dataFiltered = filter(b, a, data(:,2:7));
        dataOffset = dataFiltered - mean(dataFiltered(0.1*padFreq:0.3*padFreq,:));
        mechanics = voltToMechanics(dataOffset, scaleFactors, r, angles);
    
        subplot(subPlotNRows, subPlotNCols, trialInd)
        nRows = height(mechanics);
        plot((1:nRows)/padFreq, mechanics(:,3), 'r', LineWidth=1)
        title(sprintf("Trial No.%d  %s", trialInd, string(trialList(trialListInd, "Intensity").Variables)))
        ylim([-100 inf])
        hold on
    end
end

% Plot the first few seconds to show the synchronization peak
% halfPeakWidth = 0.4;
% nexttile
% fpXRange = syncPeakFrame(1,i)-halfPeakWidth*fpFreq:syncPeakFrame(1,i)+halfPeakWidth*fpFreq;
% plot(fpXRange/fpFreq, dataOffset(fpXRange,3), LineWidth=1)
% xlim([fpXRange(1) fpXRange(end)]/fpFreq)
% ylim([-100 max(dataOffset(fpXRange,3))+100])
% if i == 1
%     ylabel('Vertical GRF [N]')
% end
% nexttile
% padXRange = syncPeakFrame(2,i)-halfPeakWidth*fpFreq:syncPeakFrame(2,i)+halfPeakWidth*fpFreq;
% plot(padXRange/padFreq, mechanics(padXRange,3), LineWidth=1)
% xlim([padXRange(1) padXRange(end)]/padFreq)
% ylim([-100 max(mechanics(padXRange,3))+100])
% xlabel(speeds(i))
% if i == 1
%     ylabel('Boxing Pad Normal Force [N]')
% end
% xlabel(t, 'Elapsed Time [s]')
% title(t, 'Synchronization Peaks')

%% Inspect force data
data = load("D:\Overseas\German Sport University Cologne\20.Course Materials\TSM11-Project&Applied Research Methods\22.Measurement Data\S01_20260123\Qualisys\S01_20260123_Fast0001.mat");

%% Boxing pad data
subjectID = "S05";
date = "20260210";
subjectFolder = subjectID + "_" + date;
subjectDir = fullfile("..\22.Measurement Data\", subjectFolder);
padDir = fullfile(subjectDir, "Boxing Pad");
padFiles = dir(padDir);
padFiles = {padFiles.name};
for file = padFiles
    [~, fileNoExt, ext] = fileparts(file{1});
    if strcmp(ext, ".txt")
        figure()
        [b, a] = butter(2, cutOffFreq/(padFreq/2), "low");
        data = readmatrix(fullfile(padDir, file{1}));
        dataFiltered = filter(b, a, data(:,2:7));
        dataOffset = dataFiltered - mean(dataFiltered(0.1*padFreq:0.3*padFreq,:));
        mechanics = voltToMechanics(dataOffset, scaleFactors, r, angles);
        nRows = height(mechanics);
        plot((1:nRows)/padFreq, mechanics(:,3), 'r', LineWidth=1)
        ylim([-100 inf])
    end
end