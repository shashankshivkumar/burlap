package burlap.domain.singleagent.irlToolkitMDP;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import burlap.behavior.functionapproximation.dense.FromArrayFeatures;
import burlap.behavior.singleagent.Episode;
import burlap.behavior.singleagent.learning.tdmethods.QLearning;
import burlap.debugtools.RandomFactory;
import burlap.mdp.auxiliary.DomainGenerator;
import burlap.mdp.auxiliary.common.NullTermination;
import burlap.mdp.core.StateTransitionProb;
import burlap.mdp.core.TerminalFunction;
import burlap.mdp.core.action.Action;
import burlap.mdp.core.action.UniversalActionType;
import burlap.mdp.core.state.State;
import burlap.mdp.singleagent.common.VectorRF;
import burlap.mdp.singleagent.environment.Environment;
import burlap.mdp.singleagent.environment.SimulatedEnvironment;
import burlap.mdp.singleagent.model.FactoredModel;
import burlap.mdp.singleagent.model.RewardFunction;
import burlap.mdp.singleagent.model.statemodel.FullStateModel;
import burlap.mdp.singleagent.oo.OOSADomain;
import burlap.statehashing.simple.SimpleHashableStateFactory;
import burlap.domain.singleagent.gridworld.state.GridWorldState;
import burlap.domain.singleagent.irlToolkitMDP.state.irlToolkitMDPState;

public class irlToolkitMDPDomain implements DomainGenerator {
	
	private int [][][] sa_s;
	private double [][][] sa_p;
	private double [] rewardVector;
	private double [][] features;
	private int numActions;

	public irlToolkitMDPDomain(int [][][] sa_s, double [][][] sa_p, double [] rewards, double [][] features) {
		this.setTransitionFunction(sa_s, sa_p);
		this.setRewardFunction(rewards);
		this.setFeatures(features);
		this.numActions = sa_s[0].length;
	}
	
	public void setTransitionFunction(int[][][] sa_s, double[][][] sa_p) {
		this.sa_s = sa_s.clone();
		this.sa_p = sa_p.clone();
	}
	
	public void setRewardFunction(double [] rewardVector) {
		this.rewardVector = rewardVector.clone();
	}
	
	public void setFeatures(double [][] features) {
		this.features = features.clone();
	}

	
	public int[][][] getAdjacencyList() {
		int [][][] copy = new int[sa_s.length][sa_s[0].length][sa_s[0][0].length];
		for(int i = 0; i < sa_s.length; i++) {
			for (int j = 0; j < sa_s[0].length; j++) {
				for (int k = 0; k < sa_s[0][0].length; k++) {
					copy[i][j][k] = sa_s[i][j][k];
				}
			}
		}
		return copy;
	}
	
	public double[][][] getTransitionProbs() {
		double [][][] copy = new double[sa_p.length][sa_p[0].length][sa_p[0][0].length];
		for(int i = 0; i < sa_p.length; i++) {
			for (int j = 0; j < sa_p[0].length; j++) {
				for (int k = 0; k < sa_p[0][0].length; k++) {
					copy[i][j][k] = sa_p[i][j][k];
				}
			}
		}
		return copy;
	}
	
	public double[][] getFeatures() {
		double[][] copy = new double[features.length][features[0].length];
		for(int i = 0; i < features.length; i++) {
			for(int j = 0; j < features[0].length; j++) {
				copy[i][j] = features[i][j];
			}
		}
		return copy;
	}
	
	public double[] getRewardVector() {
		double[] copy = new double[rewardVector.length];
		for(int i = 0; i < rewardVector.length; i++) {
			copy[i] = rewardVector[i];
		}
		return copy;
	}
	
	public int getNumStates() {
		return sa_s.length;
	}
	
	public int getNumActions() {
		return sa_s[0].length;
	}
	
	@Override
	public OOSADomain generateDomain() {
		OOSADomain domain = new OOSADomain();
		domain.addStateClass("stateNumber", int.class);//.addStateClass("features", double.class);
			
		irlToolkitMDPModel smodel = new irlToolkitMDPModel(getAdjacencyList(), getTransitionProbs());
		RewardFunction rf = new VectorRF(getRewardVector());
		TerminalFunction tf = new NullTermination();
		
		FactoredModel model = new FactoredModel(smodel, rf, tf);
		domain.setModel(model);
		
		for(int i = 0; i < numActions; i++) {
			domain.addActionType(new UniversalActionType(Integer.toString(i)));
		}
//		domain.addActionTypes(
//				new UniversalActionType(ACTION_NORTH),
//				new UniversalActionType(ACTION_SOUTH),
//				new UniversalActionType(ACTION_EAST),
//				new UniversalActionType(ACTION_WEST));
		
		domain.addFeatureVector(features);
				
		return domain;
	}
	
	public static class irlToolkitMDPModel implements FullStateModel {
		protected int [][][] sa_s;
		protected double [][][] sa_p;
		protected Random rand = RandomFactory.getMapped(0);
		
		public irlToolkitMDPModel(int[][][] sa_s, double[][][] sa_p) {
			this.sa_s = sa_s;
			this.sa_p = sa_p;
		}
		
		@Override
		public List<StateTransitionProb> stateTransitions(State s, Action a) {
			List<StateTransitionProb> transitions = new ArrayList<StateTransitionProb>();
			int [] adjacentStates = sa_s[((irlToolkitMDPState)s).stateNumber][actionNumber(a.actionName())];
			double [] transitionProbs = sa_p[((irlToolkitMDPState)s).stateNumber][actionNumber(a.actionName())];
			
			for (int i = 0; i < adjacentStates.length; i++) {
				irlToolkitMDPState ns = new irlToolkitMDPState(adjacentStates[i]);
				double p = transitionProbs[i];
				StateTransitionProb tp = new StateTransitionProb((State)ns, p);
				transitions.add(tp);
			}
			return transitions;
		}
		
		@Override
		public State sample(State s, Action a) {
			s = s.copy();
			int [] adjacentStates = sa_s[((irlToolkitMDPState)s).stateNumber][actionNumber(a.actionName())];
			double [] transitionProbs = sa_p[((irlToolkitMDPState)s).stateNumber][actionNumber(a.actionName())];
			double roll = rand.nextDouble();
			double curSum = 0.;
			
			for (int i = 0; i < adjacentStates.length; i++) {
				curSum += transitionProbs[i];
				if(roll < curSum) {
					return (State)new irlToolkitMDPState(adjacentStates[i]);
				}
			}
			return (State)new irlToolkitMDPState(adjacentStates[0]);
		}
		
		protected int actionNumber(String name) {
			return Integer.parseInt(name);
		}
	}
	
	public static void main(String[] args) {
		int[][][] sa_s = {{{0},{1}},{{0},{2}},{{1},{2}}};
		double[][][] sa_p = {{{1},{1}},{{1},{1}},{{1},{1}}};;
		double[] rewardVector = {0,0,10};
		double[][] features = {{0,0},{0,1},{1,0}};
		irlToolkitMDPDomain mdp = new irlToolkitMDPDomain(sa_s, sa_p, rewardVector, features);
		OOSADomain domain = mdp.generateDomain();
		
		Environment env = new SimulatedEnvironment(domain, new irlToolkitMDPState(0));

    	//create a Q-learning agent
    	QLearning agent = new QLearning(domain, 0.99, new SimpleHashableStateFactory(), 1.0, 1.0);

    	//run 100 learning e0pisode and save the episode results
    	List<Episode> episodes = new ArrayList<Episode>();
    	for(int i = 0; i < 50; i++){
    		episodes.add(agent.runLearningEpisode(env,10));
    		env.resetEnvironment();
    	}
    	
    	System.out.println("Last episode: state, action");
    	for(int i = 0; i < episodes.get(episodes.size() - 1).stateSequence.size() - 1; i++) {
    		irlToolkitMDPState s = (irlToolkitMDPState)episodes.get(episodes.size() - 1).stateSequence.get(i);
    		Action a = episodes.get(episodes.size() - 1).actionSequence.get(i);
    		System.out.println(s.stateNumber + "," + a.actionName());
    	}
    	
    	FromArrayFeatures featureGen = new FromArrayFeatures(domain);
    	
    	System.out.println("State features:");
    	for(int i = 0; i < sa_s.length; i++) {
    		double[] f = featureGen.features(new irlToolkitMDPState(i));
    		for(int j = 0; j < f.length; j++) {
    			System.out.println(f[j]);
    		}
    	}
    	
    	
	}

	public int getNumFeatures() {
		return features[0].length;
	}
}
