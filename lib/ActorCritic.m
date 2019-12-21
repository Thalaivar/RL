classdef ActorCritic
    properties
        nw_
        actor_weight_
        critic_weight_
        phi_
        critic_lr_
        actor_lr_
        F2_
        F1_
        Q_
        R_
        sys_
    end
    
    methods
        function obj = ActorCritic(rl_params, dynamics)
            nw = rl_params.nw;
            obj.actor_weight_ = ones(nw,1); 
            obj.critic_weight_ = ones(nw,1);
            obj.phi_ = rl_params.regressor_func;
            obj.critic_lr_ = rl_params.critic_lr;
            obj.actor_lr_ = rl_params.actor_lr;
            obj.F1_ = rl_params.actor_F1;
            obj.F2_ = rl_params.actor_F2;
            obj.Q_ = rl_params.Q;
            obj.R_ = rl_params.R;
            obj.nw_ = nw;
            obj.sys_.f = dynamics.f;
            obj.sys_.G = dynamics.G;
            obj.sys_.nx = dynamics.nx;
        end
        
        function [u, critic_weight_update, actor_weight_update] = update_actor_critic(obj, t, x, add_noise)
            f = obj.sys_.f;
            G = obj.sys_.G;
            R = obj.R_;
            Q = obj.Q_;
            
            [u, dphi] = obj.control_policy_output(x);
            u = u + add_noise*noise(t);
            
            s = dphi*(f(x) + G(x)*u);
            
            critic_weight_update = -obj.critic_lr_*(s/(s'*s + 1)^2)*(s'*obj.critic_weight_ + Q(x) + u'*R*u);
            
            D1 = dphi*G(x)*(inv(R))*((G(x))')*dphi';
            m = s/(s'*s + 1)^2;
            s_bar = s/(s'*s + 1);
%             actor_weight_update = -obj.actor_lr_*((obj.F2_*obj.actor_weight_ - obj.F1_*s_bar'*obj.critic_weight_) - ...
%                                 0.25*D1*obj.actor_weight_*m'*obj.critic_weight_);
            
            ms = 1 + s'*s;
            actor_weight_update = -obj.actor_lr_*(0.5*D1*(obj.actor_weight_-obj.critic_weight_) - 0.25*D1*obj.actor_weight_*s'*obj.critic_weight_/ms);
        end
        
        function [u, dphi] = control_policy_output(obj, x)
            R = obj.R_;
            G = obj.sys_.G;
            
            [~, dphi] = obj.phi_(x);
            u = -0.5*(inv(R))*((G(x))')*(dphi')*obj.actor_weight_;
        end
        
        function status = check_tuning_params(obj, t, x, q, add_noise)
            f = obj.sys_.f;
            G = obj.sys_.G;
            R = obj.R_;
            nx = obj.sys_.nx;
            
            [u, dphi] = obj.control_policy_output(x);
            u = u + add_noise*noise(t);
            
            s = dphi*(f(x) + G(x)*u);
            D1 = dphi*G(x)*(inv(R))*((G(x))')*dphi';
            ms = 1 + s'*s;
            m = s/(s'*s + 1)^2;
            
            M = [q*eye(nx), zeros(nx,1), zeros(nx, obj.nw_);
                 zeros(1,nx), 1, (-0.5*obj.F1_- (1/(8*ms))*D1*obj.critic_weight_)';
                 zeros(obj.nw_, nx), (-0.5*obj.F1_- (1/(8*ms))*D1*obj.critic_weight_), obj.F2_- (1/8)*(D1*obj.critic_weight_*m' + m*obj.actor_weight_'*D1)];
            
            v = eig(M);
            min_eig = min(real(v));
            if min_eig <= 0
                status = 0;
            else
                status = 1;
            end
        end
    end
end

function n = noise(t)
    global t_prev;
    if round(t - t_prev) > 3
        n = randn(1);
    else
        n = 0;
    end
    t_prev = t;
%     n = 1*exp(-0.009*t)*(sin(t)^2*cos(t)+sin(2*t)^2*cos(0.1*t)+sin(-1.2*t)^2*cos(0.5*t)+sin(t)^5+sin(1.12*t)^2+cos(2.4*t)*sin(2.4*t)^3);
end