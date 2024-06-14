function [F]=costE(x, param)
    nimg = length(param.uv); % Number of camera poses.
    uv = param.uv;
    K = param.K;  
    
    % Extract R, T, X
    [Rvec,Tvec,X] = deserialize(x,nimg);
    nXn=0;
    for i=1:nimg
        nXn = nXn + length(uv{i}); end
    
    F = zeros(2*nXn,1); 
    
    count = 1;
    for i = 1:nimg        
        % Rotation, Translation, [X, Y, Z]
        X_idx = uv{i}(4,:); nXi = size(X_idx, 2);
        R = RotationVector_to_RotationMatrix(Rvec(:,i)); T = Tvec(:,i); Xi = X(:,X_idx);   
        for j = 1:nXi
            kp = uv{i}(1:2, j);
            
            P = K * (R * Xi(:, j) + T);
            ip = P(1:2) / P(3);

            F(count) = kp(1) - ip(1);
            F(count + 1) = kp(2) - ip(2);
            
            count = count + 2;
        end
    end
end