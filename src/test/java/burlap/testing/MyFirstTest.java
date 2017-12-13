package burlap.testing;

import burlap.behavior.functionapproximation.dense.PFFeatures;
import burlap.behavior.singleagent.Episode;
import burlap.behavior.singleagent.auxiliary.EpisodeSequenceVisualizer;
import burlap.behavior.singleagent.learnfromdemo.mlirl.MLIRL;
import burlap.behavior.singleagent.learnfromdemo.mlirl.MLIRLRequest;
import burlap.behavior.singleagent.learnfromdemo.mlirl.commonrfs.LinearStateDifferentiableRF;
import burlap.behavior.singleagent.learnfromdemo.mlirl.support.DifferentiableRF;
import burlap.behavior.singleagent.learning.tdmethods.QLearning;
import burlap.debugtools.DPrint;
import burlap.domain.singleagent.gridworld.GridWorldDomain;
import burlap.domain.singleagent.gridworld.GridWorldTerminalFunction;
import burlap.domain.singleagent.gridworld.GridWorldVisualizer;
import burlap.domain.singleagent.gridworld.state.GridLocation;
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
    	GridLocation gridloc = new GridLocation(10,10,"loc0");
    	Environment env = new SimulatedEnvironment(domain, new GridWorldState(0, 0, gridloc));

    	//create a Q-learning agent
    	QLearning agent = new QLearning(domain, 0.99, new SimpleHashableStateFactory(), 1.0, 1.0);

    	//run 100 learning episode and save the episode results
    	List<Episode> episodes = new ArrayList<Episode>();
    	for(int i = 0; i < 200; i++){
    		episodes.add(agent.runLearningEpisode(env));
    		env.resetEnvironment();
    	}
    	
    	List<Episode> goodEpisodes = new ArrayList<Episode>(episodes.subList(99, 200));
    	//visualize the completed learning episodes
    	//new EpisodeSequenceVisualizer(GridWorldVisualizer.getVisualizer(gwd.getMap()), domain, episodes);
    	
    	PFFeatures featureGen = new PFFeatures(domain);
    	LinearStateDifferentiableRF rf = new LinearStateDifferentiableRF(featureGen, 5);
    	SimpleHashableStateFactory hashingFactory = new SimpleHashableStateFactory();
    	MLIRLRequest request = new MLIRLRequest(domain, goodEpisodes, rf, hashingFactory);
    	
    	MLIRL mlirl = new MLIRL(request, .00003, .01, 500);
    	    	
    	mlirl.performIRL();
    	
    	DifferentiableRF rf_new = request.getRf();
//    	LinearStateDifferentiableRF rf_new = new LinearStateDifferentiableRF(featureGen, 5);
//    	rf_new.setParameter(0, 0);
//    	rf_new.setParameter(1,0);
//    	rf_new.setParameter(2,0);
//    	rf_new.setParameter(3,5);
//    	rf_new.setParameter(4,0);
    	
    	
    	// test learned reward
    	GridWorldDomain gwd_learned = new GridWorldDomain(11, 11);
    	gwd_learned.setMapToFourRooms();
    	gwd_learned.setTf(new GridWorldTerminalFunction(10, 10));
    	gwd_learned.setRf(rf_new);
    	OOSADomain domain_learned = gwd_learned.generateDomain();
    	Environment env_learned = new SimulatedEnvironment(domain_learned, new GridWorldState(0, 0, gridloc));

    	//create a Q-learning agent
    	QLearning agent_learned = new QLearning(domain_learned, 0.99, new SimpleHashableStateFactory(), 1.0, 1.0);

    	//run 100 learning episode and save the episode results
    	List<Episode> episodes_learned = new ArrayList<Episode>();
    	for(int i = 0; i < 200; i++){
    		episodes_learned.add(agent_learned.runLearningEpisode(env_learned));
    		env_learned.resetEnvironment();
    	}
    	
    	//visualize the completed learning episodes
    	new EpisodeSequenceVisualizer(GridWorldVisualizer.getVisualizer(gwd_learned.getMap()), domain_learned, episodes_learned);

    	
    	    	
	}
}

