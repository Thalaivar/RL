addpath('../lib/')

nw = 6;

% define parameters for Actor Critic
rl_params.nw = nw;
rl_params.regressor_func = @regressor_func;
rl_params.critic_lr = 0.1;
rl_params.actor_lr = 0.01;
rl_params.actor_F1 = ones(nw,1);
rl_params.actor_F2 = 1*eye(nw);
rl_params.Q = @Q;
rl_params.R = 1;

% dynamics of system
nx = 3;
dynamics.f = @linear_plant_f;
dynamics.G = @linear_plant_G;
dynamics.nx = nx;

ac = ActorCritic(rl_params, dynamics);

tspan = [0, 1000];
x0 = [1; 1; 1];
options = odeset('OutputFcn', @odeplot);
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
legend('$x_1$', '$x_2$', '$x_3$', 'Interpreter', 'latex')
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

function Qx = Q(x)
    Qx = x'*x;
end

function [phi, dphi] = regressor_func(x)
    x1 = x(1,1); x2 = x(2,1); x3 = x(3,1);

    phi = [x1^2; x1*x2; x2^2; x3^2; x1*x3; x2*x3];
    dphi = [2*x1,    0,    0; 
              x2,   x1,    0; 
               0, 2*x2,    0;
               0,    0, 2*x3;
              x3,    0,   x1;
               0,   x3,   x2]; 
end

function f = linear_plant_f(x)
    A = [-1.01887,  0.90506, -0.00215;
      0.82225, -1.07741, -0.17555;
      0      ,  0      , -1];
    f = A*x;
end

function G = linear_plant_G(x)
    G = [0;0;1];
end

function f = nonlinear_plant_f(x)
    x1 = x(1); x2 = x(2);
    f = [-x1 + x2;
         -0.5*x1 - 0.5*x2*(1 - (cos(2*x1) + 2)^2)];
end

function G = nonlinear_plant_G(x)
    x1 = x(1);
    G = [0; cos(2*x1) + 2];
end