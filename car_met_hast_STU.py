def car_lik(parameters,t,x,error_vars):

    error_vars = error_vars**2

    x = x.astype('float64')
    t = t.astype('float64')
    error_vars = error_vars.astype('float64')

    sigma = parameters[0]
    tau = parameters[1]

    #b = parameters[1] #comment it to do 2 pars estimation
    #tau = params(1,1);
    #sigma = sqrt(2*var(x)/tau);

    b = np.mean(x)/tau

    epsilon = 1e-300
    cte_neg = -np.infty
    num_datos = np.size(x)

    Omega = []
    x_hat = []
    a = []
    x_ast = []

    Omega.append((tau*(sigma**2))/2.)
    x_hat.append(0.)
    a.append(0.)
    x_ast.append(x[0] - b*tau)

    loglik = 0.

    for i in range(1,num_datos):

        a_new = np.exp(-(t[i]-t[i-1])/tau)
        x_ast.append(x[i] - b*tau)
        x_hat.append(a_new*x_hat[i-1] + (a_new*Omega[i-1]/(Omega[i-1] + error_vars[i-1]))*(x_ast[i-1]-x_hat[i-1]))

        Omega.append(Omega[0]*(1-(a_new**2)) + ((a_new**2))*Omega[i-1]*( 1 - (Omega[i-1]/(Omega[i-1]+ error_vars[i-1]))))

        loglik_inter = -0.5*( np.log(2*np.pi*(Omega[i] + error_vars[i])) + ((x_hat[i]-x_ast[i])**2) / (Omega[i] + error_vars[i]))
                       #+ np.log(epsilon)

        loglik = loglik + loglik_inter

    return loglik
