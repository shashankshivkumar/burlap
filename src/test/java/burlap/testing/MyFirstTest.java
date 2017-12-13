package burlap.testing;

import burlap.behavior.functionapproximation.dense.PFFeatures;
import burlap.behavior.singleagent.Episode;
import burlap.behavior.singleagent.auxiliary.EpisodeSequenceVisualizer;
import burlap.behavior.singleagent.learnfromdemo.mlirl.MLIRL;
import burlap.behavior.singleagent.learnfromdemo.mlirl.MLIRLRequest;
import burlap.behavior.singleagent.learnfromdemo.mlirl.commonrfs.LinearStateDifferentiableRF;
import burlap.behavior.singleagent.learning.tdmethods.QLearning;
import burlap.domain.singleagent.gridworld.GridWorldDomain;
import burlap.domain.singleagent.gridworld.GridWorldTerminalFunction;
import burlap.domain.singleagent.gridworld.GridWorldVisualizer;
import burlap.domain.singleagent.gridworld.state.GridWorldState;
import burlap.statehashing.simple.SimpleHashableStateFactory;

import burlap.mdp.singleagent.SADomain;
import burlap.mdp.singleagent.environment.Environment;
import burlap.mdp.singleagent.environment.SimulatedEnvironment;
import burlap.mdp.singleagent.oo.OOSADomain;

import java.util.ArrayList;
import java.util.List;


public class MyFirstTest {
	public static void main(String[] args) {
    	//define the problem
    	GridWorldDomain gwd = new GridWorldDomain(11, 11);
    	gwd.setMapToFourRooms();
    	gwd.setTf(new GridWorldTerminalFunction(10, 10));
    	OOSADomain domain = gwd.generateDomain();
    	Environment env = new SimulatedEnvironment(domain, new GridWorldState(0, 0));

    	//create a Q-learning agent
    	QLearning agent = new QLearning(domain, 0.99, new SimpleHashableStateFactory(), 1.0, 1.0);

    	//run 100 learning episode and save the episode results
    	List<Episode> episodes = new ArrayList<Episode>();
    	for(int i = 0; i < 100; i++){
    		episodes.add(agent.runLearningEpisode(env));
    		env.resetEnvironment();
    	}

    	//visualize the completed learning episodes
    	//new EpisodeSequenceVisualizer(GridWorldVisualizer.getVisualizer(gwd.getMap()), domain, episodes);
    	
    	PFFeatures featureGen = new PFFeatures(domain);
    	LinearStateDifferentiableRF rf = new LinearStateDifferentiableRF(featureGen, 6);
    	SimpleHashableStateFactory hashingFactory = new SimpleHashableStateFactory();
    	MLIRLRequest request = new MLIRLRequest(domain, episodes, rf, hashingFactory);
    	
    	MLIRL mlirl = new MLIRL(request, .0003, .0001, 100);
    	
    	mlirl.performIRL();
    	    	
	}
}

