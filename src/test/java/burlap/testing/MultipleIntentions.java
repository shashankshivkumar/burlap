package burlap.testing;

import burlap.behavior.functionapproximation.dense.PFFeatures;
import burlap.behavior.singleagent.Episode;
import burlap.behavior.singleagent.auxiliary.EpisodeSequenceVisualizer;
import burlap.behavior.singleagent.learnfromdemo.mlirl.MLIRL;
import burlap.behavior.singleagent.learnfromdemo.mlirl.MLIRLRequest;
import burlap.behavior.singleagent.learnfromdemo.mlirl.MultipleIntentionsMLIRL;
import burlap.behavior.singleagent.learnfromdemo.mlirl.MultipleIntentionsMLIRLRequest;
import burlap.behavior.singleagent.learnfromdemo.mlirl.commonrfs.LinearStateDifferentiableRF;
import burlap.behavior.singleagent.learnfromdemo.mlirl.support.DifferentiableRF;
import burlap.behavior.singleagent.learning.tdmethods.QLearning;
import burlap.debugtools.DPrint;
import burlap.domain.singleagent.gridworld.GridWorldDomain;
import burlap.domain.singleagent.gridworld.GridWorldTerminalFunction;
import burlap.domain.singleagent.gridworld.GridWorldVisualizer;
import burlap.domain.singleagent.gridworld.state.GridAgent;
import burlap.domain.singleagent.gridworld.state.GridLocation;
import burlap.domain.singleagent.gridworld.state.GridWorldState;
import burlap.statehashing.simple.SimpleHashableStateFactory;
import burlap.mdp.singleagent.SADomain;
import burlap.mdp.singleagent.environment.Environment;
import burlap.mdp.singleagent.environment.SimulatedEnvironment;
import burlap.mdp.singleagent.model.FactoredModel;
import burlap.mdp.singleagent.oo.OOSADomain;

import java.util.ArrayList;
import java.util.List;

public class MultipleIntentions {
	public static void main(String[] args) {
    	//define the domains
		int numEnvs = 4;
		int gridSize = 11;
		int [] terminalStateX = {10, 6, 0, 10};
		int [] terminalStateY = {10, 5, 10, 0};
		int initStateX = 0;
		int initStateY = 0;
		List<GridWorldDomain> gwds = new ArrayList<GridWorldDomain>();
		List<OOSADomain> domains = new ArrayList<OOSADomain>();
		List<Environment> envs = new ArrayList<Environment>();
		List<QLearning> agents = new ArrayList<QLearning>();
		List<PFFeatures> featureGens = new ArrayList<PFFeatures>();
		List<LinearStateDifferentiableRF> rfs = new ArrayList<LinearStateDifferentiableRF>();
		FactoredModel model_tmp;
		List<GridLocation> gridlocs = new ArrayList<GridLocation>();
		ArrayList<Episode> goodEpisodes = new ArrayList<Episode>();
		int numTrajsPerTask = 20;
		
		for(int i = 0; i < numEnvs; i++)
		{
			gridlocs.add(new GridLocation(terminalStateX[i], terminalStateY[i], "loc" + String.valueOf(i)));
		}
		for(int i = 0; i < numEnvs; i++)
		{
			gwds.add(new GridWorldDomain(gridSize, gridSize));
			gwds.get(i).setMapToFourRooms();
//			gwds.get(i).setTf(new GridWorldTerminalFunction(terminalStateX[i], terminalStateY[i]));
			domains.add(gwds.get(i).generateDomain());
			featureGens.add(new PFFeatures(domains.get(i)));
			rfs.add(new LinearStateDifferentiableRF(featureGens.get(i), 4+numEnvs));
			for(int j = 0; j < 4+numEnvs; j++)
			{
				if (j == i+3) rfs.get(i).setParameter(j, 10);
				else rfs.get(i).setParameter(j, 0);
			}
			gwds.get(i).setRf(rfs.get(i));
			model_tmp = (FactoredModel)domains.get(i).getModel();
			model_tmp.setRf(rfs.get(i));
			domains.get(i).setModel(model_tmp);
			
			envs.add(new SimulatedEnvironment(domains.get(i), new GridWorldState(new GridAgent(initStateX, initStateY), gridlocs)));
			agents.add(new QLearning(domains.get(i), 0.99, new SimpleHashableStateFactory(), 1.0, 1.0));
			List<Episode> episodes = new ArrayList<Episode>();
			for(int j = 0; j < 200; j++){
				episodes.add(agents.get(i).runLearningEpisode(envs.get(i), 200));
	    		envs.get(i).resetEnvironment();
	    	}
			
			for(int j = 0; j < numTrajsPerTask; j++) {
				goodEpisodes.add(agents.get(i).runLearningEpisode(envs.get(i), 50));
				envs.get(i).resetEnvironment();
			}
		}
		
		// Visualize demo episodes	
		// new EpisodeSequenceVisualizer(GridWorldVisualizer.getVisualizer(gwds.get(0).getMap()), domains.get(0), goodEpisodes);
		
		// Start Learning Multiple Intentions
		GridWorldDomain gwd = new GridWorldDomain(gridSize, gridSize);
    	gwd.setMapToFourRooms();
    	OOSADomain domain = gwd.generateDomain();
    	Environment env = new SimulatedEnvironment(domain, new GridWorldState(new GridAgent(initStateX, initStateY), gridlocs));
    	PFFeatures featureGen = new PFFeatures(domain);
    	LinearStateDifferentiableRF rf = new LinearStateDifferentiableRF(featureGen, 4+numEnvs);
    	SimpleHashableStateFactory hashingFactory = new SimpleHashableStateFactory();
    	int numClusters = 4;
    	MultipleIntentionsMLIRLRequest MIRequest = new MultipleIntentionsMLIRLRequest(domain, goodEpisodes, rf, numClusters, hashingFactory);
    	
    	int numEMSteps = 20;
    	double learningRate = 0.0003;
    	double maxMLIRLLikelihoodChange = 0.01;
    	int maxMLIRLSteps = 50;
    	MultipleIntentionsMLIRL MI = new MultipleIntentionsMLIRL(MIRequest, numEMSteps, learningRate, maxMLIRLLikelihoodChange, maxMLIRLSteps, numTrajsPerTask);
    	MI.performIRL();
    	    	
    	// Test Learned Rewards
		List<DifferentiableRF> rfs_learned = MI.getClusterRFs();		
		List<GridWorldDomain> gwds_learned = new ArrayList<GridWorldDomain>();
		List<OOSADomain> domains_learned = new ArrayList<OOSADomain>();
		List<Environment> envs_learned = new ArrayList<Environment>();
		List<QLearning> agents_learned = new ArrayList<QLearning>();
		List<Episode> goodEpisodesLearned = new ArrayList<Episode>();
    	for(int i = 0; i < numClusters; i++)
    	{
			gwds_learned.add(new GridWorldDomain(gridSize, gridSize));
    		gwds_learned.get(i).setMapToFourRooms();
	    	gwds_learned.get(i).setRf(rfs_learned.get(i));
	    	domains_learned.add(gwds_learned.get(i).generateDomain());
	    	envs_learned.add(new SimulatedEnvironment(domains_learned.get(i), new GridWorldState(new GridAgent(initStateX, initStateY), gridlocs)));
	
	    	//create a Q-learning agent
	    	agents_learned.add(new QLearning(domains_learned.get(i), 0.99, new SimpleHashableStateFactory(), 1.0, 1.0));
	    	
	    	//run 100 learning episode and save the episode results
	    	List<Episode> episodesLearned = new ArrayList<Episode>();
	    	for(int j = 0; j < 200; j++){
	    		episodesLearned.add(agents_learned.get(i).runLearningEpisode(envs_learned.get(i), 200));
	    		envs_learned.get(i).resetEnvironment();
	    	}
	    	
	    	for(int j = 0; j < 20; j++) {
				goodEpisodesLearned.add(agents_learned.get(i).runLearningEpisode(envs_learned.get(i), 50));
				envs_learned.get(i).resetEnvironment();
			}
    	}
    	
    	//visualize episodes from learned rewards
    	new EpisodeSequenceVisualizer(GridWorldVisualizer.getVisualizer(gwds_learned.get(0).getMap()), domains_learned.get(0), goodEpisodesLearned);
	}
}
