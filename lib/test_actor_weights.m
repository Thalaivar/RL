function test_actor_weights(ac, tfin, x0)
    tspan = [0, tfin];
    [t, y] = ode45(@(t,y) test_actor_simulate(t, y, ac), tspan, x0);
    plot(t, y);
end

function dy = test_actor_simulate(t, y, ac)
    [u,~] = ac.control_policy_output(y);
    dy = ac.sys_.f(y) + ac.sys_.G(y)*u;
end