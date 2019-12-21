function G = inverted_pendulum_dynamics_G(x)
    phi = x(3);
    
    mp = 1; M = 5; m = mp + M;
    l = 1; R = 0.025;
    I3 = 0.25*mp*R^2 + (mp*l^2)/12;
    
    G = [0;
         (4*I3 + mp*l^2)/((4*I3 + mp*l^2)*m - (mp*l*cos(phi))^2);
         0;
         -4*mp*l*cos(phi)/(8*I3*m + mp*(mp + 2*M)*l^2 - ((mp*l)^2)*cos(2*phi))];
end