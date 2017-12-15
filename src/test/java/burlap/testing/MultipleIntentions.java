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
		ArrayList<ArrayList<Episode>> allTasksEpisodes = new ArrayList<ArrayList<Episode>>();
		List<PFFeatures> featureGen = new ArrayList<PFFeatures>();
		List<LinearStateDifferentiableRF> rfs_learned = new ArrayList<LinearStateDifferentiableRF>();
		List<SimpleHashableStateFactory> hashingFactories = new ArrayList<SimpleHashableStateFactory>();
		List<LinearStateDifferentiableRF> rfs = new ArrayList<LinearStateDifferentiableRF>();
		FactoredModel model_tmp;
		List<GridLocation> gridlocs = new ArrayList<GridLocation>();
		ArrayList<Episode> goodEpisodes = new ArrayList<Episode>();
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
			featureGen.add(new PFFeatures(domains.get(i)));
			rfs.add(new LinearStateDifferentiableRF(featureGen.get(i), 4+numEnvs));
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
			
			for(int j = 0; j < 20; j++) {
				goodEpisodes.add(agents.get(i).runLearningEpisode(envs.get(i), 50));
				envs.get(i).resetEnvironment();
			}
		}
		
		// Visualize		
		new EpisodeSequenceVisualizer(GridWorldVisualizer.getVisualizer(gwds.get(0).getMap()), domains.get(0), goodEpisodes);
	}
}

