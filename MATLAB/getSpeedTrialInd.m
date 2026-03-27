function [speed, speedInd, trialInd] = getSpeedTrialInd(fileNoExt, instrument)
%GETSPEEDTRIALIND extracts the trial index from the name of recorded
% measurement file. The corresponding valid trial index needs to be paired
% otherwise.
    speeds = ["Slow", "Normal", "Fast", "Static"];
    nameSplit = split(fileNoExt, '_');
    speedTrialInd = char(nameSplit{3});

    if strcmp(instrument, "Qualisys")
        speed = speedTrialInd(1:end-4);
        trialInd = str2double(speedTrialInd(end-3:end));
    elseif strcmp(instrument, "Boxing Pad")
        speed = speedTrialInd(1:end-2);
        trialInd = str2double(speedTrialInd(end-1:end));
    else
        error(fprintf("The instrument %s is not supported.", instrument))
    end

    speedInd = find(speed == speeds);
    if isempty(speedInd)
        speedInd = length(speeds) + 1;
        fprintf("The speed %s is not included in the experiment " + ...
            "configuration. An extra index %d is assigned instead.\n", ...
            speed, speedInd);
    end
end