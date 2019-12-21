addpath('../lib/')

nw = 14;

% define parameters for Actor Critic
rl_params.nw = nw;
rl_params.regressor_func = @regressor_func;
rl_params.critic_lr = 1;
rl_params.actor_lr = 1;
% rl_params.actor_F1 = 0.001*ones(nw,1);
% rl_params.actor_F2 = 100*eye(nw);
rl_params.Q = @Q;
rl_params.R = 1;

% dynamics of system
nx = 4;
dynamics.f = @inv_pendulum_f;
dynamics.G = @inv_pendulum_G;
dynamics.nx = nx;

ac = ActorCritic(rl_params, dynamics);

tspan = [0, 1000];
x0 = [1; 0; 0; 0];
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
% test_actor_weights(ac, 1000, x0);

rmpath('../lib/')

function qx = Q(x)
    if x(3) ~= 0 && mod(x(3), 2*pi) == 0
        x(3) = 2*pi;
    else
        x(3) = mod(x(3), 2*pi);
    end
    qx = x'*diag([0.1, 0.1, 10, 10])*x;
end

function [phi, dphi] = regressor_func(x)
    r = x(1); rdot = x(2);
    ph = x(3); phdot = x(4);
    
    phi = [rdot^2; ph^2; phdot^2; r^2; r*rdot; r*ph; r*phdot; rdot*ph; rdot*phdot; phdot*ph; ... 
           cos(ph)*sin(ph); sin(ph)*phdot^2; sin(ph); sin(ph)*cos(ph)*phdot^2];
    dphi = [0, 2*rdot, 0, 0;
            0, 0, 2*ph, 0;
            0, 0, 0, 2*phdot;
            2*r, 0, 0, 0;
            rdot, r, 0, 0;
            ph, 0, r, 0;
            phdot, 0, 0, r;
            0, ph, rdot, 0;
            0, phdot, 0, rdot;
            0, 0, phdot, ph;
            0, 0, cos(2*ph), 0;
            0, 0, cos(ph)*phdot^2, 2*sin(ph)*phdot;
            0, 0, cos(ph), 0;
            0, 0, cos(2*ph)*phdot^2, sin(ph)*cos(ph)*2*phdot];
end

function f = inv_pendulum_f(x)
    m = 0.15; mc = 1; l = 0.75; g = 9.8; M = mc + m;
    r = x(1); rdot = x(2); phi = x(3); phidot = x(4);
    
    f = [rdot;
         (-m*g*cos(phi)*sin(phi)+m*l*sin(phi)*phidot^2)/(M - m*(cos(phi))^2);
         phidot;
         -(-M*g*sin(phi) + m*l*cos(phi)*sin(phi)*phidot^2)/(M*l - m*l*(cos(phi))^2)];         
end

function G = inv_pendulum_G(x)
    m = 0.15; mc = 1; l = 0.75; g = 9.8; M = mc + m;
    r = x(1); rdot = x(2); phi = x(3); phidot = x(4);
    G = [0;
         1/(M - m*(cos(phi))^2);
         0;
         -cos(phi)/(M*l - m*l*(cos(phi))^2)];
end