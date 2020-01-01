clearvars
close all

env = oscillatoryCartPole;

obsInfo = getObservationInfo(env);
actInfo = getActionInfo(env);

nx = obsInfo.Dimension(1);
nu = actInfo.Dimension(1);

% critic network
statePath = [imageInputLayer([nx 1], 'Normalization', 'none', 'Name', obsInfo.Name)
             fullyConnectedLayer(300, 'Name', 'CriticFC1')
             reluLayer('Name', 'CriticReLU1')
             fullyConnectedLayer(400, 'Name', 'CriticFC2')];
actionPath = [imageInputLayer([nu 1], 'Normalization', 'none', 'Name', 'action')
              fullyConnectedLayer(400, 'Name', 'CriticFC3')];
commonPath = [additionLayer(2, 'Name', 'CriticAdd')
              reluLayer('Name', 'CriticReLU2')
              fullyConnectedLayer(1, 'Name', 'criticOut')];

criticNetwork = layerGraph(statePath);
criticNetwork = addLayers(criticNetwork, actionPath);
criticNetwork = addLayers(criticNetwork, commonPath);

criticNetwork = connectLayers(criticNetwork, 'CriticFC2', 'CriticAdd/in1');
criticNetwork = connectLayers(criticNetwork, 'CriticFC3', 'CriticAdd/in2');

criticOpts = rlRepresentationOptions('LearnRate', 1e-2, 'GradientThreshold', 1);
critic = rlRepresentation(criticNetwork, obsInfo, actInfo, 'Observation', {'states'}, 'Action', {'action'}, criticOpts);

% actor network
statePath = [
    imageInputLayer([nx 1],'Normalization','none','Name', obsInfo.Name)
    fullyConnectedLayer(300,'Name','ActorFC1')
    reluLayer('Name','ActorRelu1')
    fullyConnectedLayer(400,'Name','ActorFC2')
    reluLayer('Name','ActorRelu2')
    fullyConnectedLayer(nu,'Name','ActorFC3')
    tanhLayer('Name','ActorTanh1')
    scalingLayer('Name','actorOut','Scale',max(actInfo.UpperLimit))];
actorNetwork = layerGraph(statePath);

actorOpts = rlRepresentationOptions('LearnRate', 1e-3, 'GradientThreshold', 1);
actor = rlRepresentation(actorNetwork, obsInfo, actInfo, 'Observation', {'states'}, 'Action', {'actorOut'}, actorOpts);

% create agent
agentOpts = rlDDPGAgentOptions(...
            'SampleTime', env.Ts,...
            'TargetSmoothFactor',1e-3,...
            'ExperienceBufferLength',1e6,...
            'MiniBatchSize',128);
agentOpts.NoiseOptions.Variance = 1.5;
agentOpts.NoiseOptions.VarianceDecayRate = 1e-7;

agent = rlDDPGAgent(actor,critic,agentOpts);
% load('cartPoleDDPG.mat', 'agent');

plot(env);
maxepisodes = 1e4;
maxsteps = 1e6;
trainOpts = rlTrainingOptions(...
    'MaxEpisodes', maxepisodes, ...
    'MaxStepsPerEpisode', maxsteps, ...
    'Verbose', false, ...
    'Plots','training-progress',...
    'StopTrainingCriteria','AverageSteps',...
    'StopTrainingValue', 60/env.Ts,...
    'ScoreAveragingWindowLength',10); 
trainingStats = train(agent,env,trainOpts);

save('cartPoleDDPG.mat', 'agent');