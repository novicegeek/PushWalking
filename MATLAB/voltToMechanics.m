function outMechanics1 = voltToMechanics(voltData, scaleFactors, r, angles)
    a1 = angles(1);
    a2 = angles(2);
    a3 = angles(3);
    % Use row vector only
    if width(scaleFactors) == 1
        scaleFactors = scaleFactors';
    end

    % Scale the raw voltage data
    scaledVolt1 = voltData(:,1:6) .* scaleFactors(1:6);

    % Create conversion matrix
    conversionMat1 = zeros(6);
    conversionMat1(:,1) = [
        cosd(a1), -sind(a3), -sind(a3), ...
        cosd(a1), -sind(a3), -sind(a3)]';
    conversionMat1(:,2) = cosd(a1)*cosd(a3)*[0, -1, 1, 0, -1, 1]';
    conversionMat1(:,3) = sind(a1)*ones(6, 1);
    conversionMat1(:,4) = r*[
        -sind(a1), -cosd(a2), cosd(a2), ...
        sind(a1), cosd(a2), -cosd(a2)]';
    conversionMat1(:,5) = r*sind(a2)*sind(a1)*[0, 1, 1, 0, -1, -1]';
    conversionMat1(:,6) = r*cosd(a1)*[-1, 1, -1, 1, -1, 1]';

    % Convert to force
    outMechanics1 = scaledVolt1 * conversionMat1;
end