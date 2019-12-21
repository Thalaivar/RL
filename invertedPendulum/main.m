addpath('../lib/')

nw = 10;

% define parameters for Actor Critic
rl_params.nw = nw;
rl_params.regressor_func = @regressor_func;
rl_params.critic_lr = 0.1;
rl_params.actor_lr = 1;
% rl_params.actor_F1 = 0.001*ones(nw,1);
% rl_params.actor_F2 = 100*eye(nw);
rl_params.Q = @Q;
rl_params.R = 1;

% dynamics of system
nx = 4;
dynamics.f = @inverted_pendulum_dynamics_f;
dynamics.G = @inverted_pendulum_dynamics_G;
dynamics.nx = nx;

ac = ActorCritic(rl_params, dynamics);

tspan = [0, 800];
x0 = [0; 0; pi; 0];
options = odeset('OutputFcn', @odeplot);
global t_prev;
t_prev = 0;
[t,y] = ode15s(@(t,y) simulate_system(t, y, ac), tspan, [x0;ac.critic_weight_;ac.actor_weight_], options); 

% plot results
x_time_history = y(:,1:nx);
W1_time_history = y(:,nx+1:nx+nw);
W2_time_history = y(:,nx+nw+1:2*nw+nx);

actor_weight_labels = {};
critic_weight_labels = {};
for i = 1:nw
    actor_weight_labels{i} = strcat('$W_{a_', num2str(i), '}$');
    critic_weight_labels{i} = strcat('$W_{c_', num2str(i), '}$');
end

figure
plot(t, x_time_history)
legend('$r$', '$\dot{r}$', '$\phi$', '$\dot{\phi}$', 'Interpreter', 'latex')
title('States vs. time')
xlabel('$t(s)$', 'Interpreter', 'latex')
ylabel('$x_i(t)$', 'Interpreter', 'latex');

figure
plot(t, W1_time_history)
legend(critic_weight_labels, 'Interpreter', 'latex')
title('Critic weights vs. time')
xlabel('$t(s)$', 'Interpreter', 'latex')

figure
plot(t, W2_time_history)
legend(actor_weight_labels, 'Interpreter', 'latex')
title('Actor weights vs. time')
xlabel('$t(s)$', 'Interpreter', 'latex')

V_time_history = zeros(length(t),1);

for i = 1:length(t)
    [phi,~] = ac.phi_(x_time_history(i,:)');
    V_time_history(i) = W1_time_history(i,:)*phi;
end
figure
plot(t, V_time_history)
title('$\hat{V}(\mathbf{x})$ vs. $t$', 'Interpreter', 'latex')
xlabel('$t(s)$', 'Interpreter', 'latex')

% assign converged weights to actor-critic
ac.actor_weight_ = W2_time_history(end,:)';
ac.critic_weight_ = W1_time_history(end,:)';

% test results
test_actor_weights(ac, 1000, x0);

rmpath('../lib/')

function qx = Q(x)
    qx = x'*diag([1000, 10, 0.1, 5])*x;
end

function [phi, dphi] = regressor_func(x)
    x1 = x(1); x2 = x(2);
    x3 = x(3); x4 = x(4);
    phi = [x1^2; x2^2; x3^2; x4^2; x1*x2; x1*x3; x1*x4; x2*x3; x2*x4; x3*x4; sin(x3)*x4^2];
    dphi = [2*x1,0,0,0;
            0,2*x2,0,0;
            0,0,2*x3,0;
            0,0,0,2*x4;
            x2,x1,0,0;
            x3,0,x1,0;
            x4,0,0,x1;
            0,x3,x2,0;
            0,x4,0,x2;
            0,0,x4,x3;
            0,0,cos(x3)*x4^2,2*x4*sin(x3)];
end