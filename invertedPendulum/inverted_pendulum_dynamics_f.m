function f = inverted_pendulum_dynamics_f(x, platform_dynamics)
    r = x(1); rdot = x(2);
    phi = x(3); phidot = x(4);
    
    mp = 0.1; M = 1; m = mp + M;
    l = 1; R = 0.025;
    I3 = 0.25*mp*R^2 + (mp*l^2)/12;
    
%     [th, thdot, thddot] = platform_dynamics(t);
    th = 0; thdot = 0; thddot = 0;
    
    g = 9.8;
    
    rddot_f = ((4*I3 + mp*l^2)*(- mp*r*thdot^2 - 0.5*mp*l*sin(phi)*(thdot - phidot)^2 ...
            - 0.5*mp*l*cos(phi)*thddot) + (mp*l/2)*cos(phi)*(-2*mp*g*l*sin(th - phi) ...
            + 4*mp*l*sin(phi)*rdot*thdot + 4*I3*thddot + mp*l^2*thddot + 2*mp*l*r*(cos(phi)*thdot^2 ...
            + sin(phi)*thddot)))/(-(4*I3 + mp*l^2)*m + ((mp*l*cos(phi))^2));
        
    phiddot_f = (-4*(mp^2)*g*l*sin(th - phi) - 4*mp*M*g*l*sin(th - phi) ...
        + 8*mp*m*l*sin(phi)*rdot*thdot - ((mp*l)^2)*sin(2*phi)*thdot^2 ... 
        + 2*((mp*l)^2)*sin(2*phi)*thdot*phidot - ((mp*l)^2)*sin(2*phi)*phidot^2 ... 
        + 8*I3*mp*thddot + ((mp*l)^2 + 8*I3*M)*thddot + 2*mp*M*(l^2)*thddot ... 
        - ((mp*l)^2)*cos(2*phi)*thddot + 4*mp*l*r*(M*cos(phi)*thdot^2 ... 
        + m*sin(phi)*thddot))/(8*I3*m + mp*(mp + 2*M)*l^2 - ((mp*l)^2)*cos(2*phi));
    
    f = [rdot; rddot_f; phidot; phiddot_f];
end