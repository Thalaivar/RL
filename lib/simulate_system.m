function dy = simulate_system(t, y, ac)
    nx = ac.sys_.nx;
    x = y(1:nx);
    
    nw = ac.nw_;
    ac.critic_weight_ = y(nx+1:nx+nw,1);
    ac.actor_weight_ = y(nx+nw+1:nx+2*nw,1);
    
    add_noise = 1;
    
%     if ac.check_tuning_params(t, x, 0.5, add_noise) == 0
%         warning('Tuning params check failed!\n');
%     end
    
    [u, critic_weight_update, actor_weight_update] = ac.update_actor_critic(t, x, add_noise);
    
    dx = ac.sys_.f(x) + ac.sys_.G(x)*u;
    
    dy = [dx; critic_weight_update; actor_weight_update];
end