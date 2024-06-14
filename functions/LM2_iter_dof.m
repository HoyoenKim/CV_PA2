function [x_BA] = LM2_iter_dof(x, param)
    x=transpose(x);
    disp(['Size of x: ', mat2str(size(x))]);
    
    iter = 0;
    tolX = 5e-3;   % x�� ����������
    tolFun = 5e-3; % �Լ����� ����������
    tolJ = 1e-5;   % Jacobian�� ����������
    max_iter = 50;
    max_lambda = 1
    x_BA = cell(0,1);
    
    nimgs = length(param.uv); nX = param.nX; %length(param.X_img);
    key1 = param.key1; key2 = param.key2;
    opt = param.optimization; dof = param.dof_remove;
    %dof_idx = [ 6*key1-5 : 6*key1];
    dof_idx = [ 6*key1-5 : 6*key1, 6*key2-2];
%    max_X = max(param.X_counts);
    while iter < max_iter
        disp(['iter: ', mat2str(iter)]);
        iter = iter+1;        
        
        [F,J] = JacobiancostE(x, param); 
        if dof == 1, J(:,dof_idx) = []; end
        
        before_cost = norm(F)^2;
        
        H = J'*J;
        JtF = J'*F;
        
        if opt==1      % Levenberg - Marquarbt
            if iter==1, lambda = 1e-3; end
            H_LM = H + lambda*sparse(diag(diag(H)));            
            %H_LM = H + sparse(1:length(x),1:length(x),lambda,length(x),length(x));
        elseif opt==0  % Levenberg
            if iter==1, lambda = 1e-3*mean(diag(H)); end
            H_LM = H + lambda*eye(size(H,1));                   
        end   
                
        dp = -H_LM \ JtF;
        if isnan(sum(dp)) % NaN�̸� �׳� ������
            break;
%             lambda=lambda*10; disp(['lambda : ', num2str(lambda), 'by NaN']);
%             if opt==1      % Levenberg - Marquarbt
%                 H_LM = H + lambda*sparse(diag(diag(H)));
%                 dp = -H_LM \ JtF;
%             elseif opt==0  % Levenberg
%                 H_LM = H + lambda*eye(size(H,1));
%                 dp = -H_LM \ JtF;
%             end  
        end
%         ratio = repmat(max_X,1,length(param.X_counts))./param.X_counts;
%         ratio = reshape(repmat(ratio,3,1),[],1);
%         dp(6*nimgs+1:end) = dp(6*nimgs+1:end) .* ratio;
        if dof == 1
            dp_idx = 1:6*nimgs+3*nX; dp_idx(dof_idx) = [];
            dp_temp = zeros(6*nimgs+3*nX,1); dp_temp(dp_idx) = dp; 
            x_LM = x+dp_temp;
        elseif dof == 0
            x_LM = x+dp;
        end
        
        after_F=costE(x_LM, param);
        after_cost = norm(after_F)^2;
        l2_p  = norm(x);
        l2_dp = norm(dp); %dp�� ��ȭ��    
            
        if (after_cost < before_cost)
            lambda=lambda*0.1; disp(['lambda : ', num2str(lambda)]);
            x=x_LM;
        else
            while after_cost >= before_cost
                lambda=lambda*10; disp(['lambda : ', num2str(lambda)]);
                if lambda > max_lambda
                    break;
                end
                
                if opt==1,     H_LM = H + lambda*sparse(diag(diag(H)));  % Levenberg - Marquarbt
                elseif opt==0, H_LM = H + lambda*eye(size(H,1));     % Levenberg
                end          % H_LM = H + sparse(1:length(x),1:length(x),lambda,length(x),length(x)); % Levenberg;                
                
                dp = -H_LM \ JtF;
%                 ratio = repmat(max_X,1,length(param.X_counts))./param.X_counts;
%                 ratio = reshape(repmat(ratio,3,1),[],1);
%                 dp(6*nimgs+1:end) = dp(6*nimgs+1:end) .* ratio; 
                
                if dof == 1
                    dp_idx = 1:6*nimgs+3*nX; dp_idx(dof_idx) = [];
                    dp_temp = zeros(6*nimgs+3*nX,1);  dp_temp(dp_idx) = dp;
                    x_LM = x+dp_temp;
                elseif dof == 0
                    x_LM = x+dp;
                end
                
                after_F=costE(x_LM, param);
                after_cost = norm(after_F)^2;
            end
            x = x_LM;
        end        
        disp(['Iteration: ',num2str(iter),'   Cost value is ',num2str(before_cost)]);
        
        %if ( abs(before_cost - after_cost) <= tolFun )        
        if ( abs(before_cost - after_cost) < tolFun*before_cost )
            disp('Finished (tolFun)');
            x_BA{end+1} = x_LM;
            break;
        end
        
        if ( l2_dp < tolX*l2_p )
            disp('Finished (tolX)');
            x_BA{end+1} = x_LM;
            break;
        end
                
        if mod(iter,5)==0 || iter<10
            x_BA{end+1} = x_LM;
        end
        
%         X3d = dp(6*nimgs+1:end);
%         dX_dist = zeros(1,length(param.X_counts));
%         for i = 1:length(param.X_counts)
%             dX_dist(i) = norm(X3d(3*i-2:3*i));
%         end
%         f = figure; movegui(f,[2000 800]);
%         scatter(param.X_counts, dX_dist);
%         close;
    end

    if iter==max_iter
        x_BA{end+1} = x_LM;
        disp('Finished (max_iter)');
    end

end