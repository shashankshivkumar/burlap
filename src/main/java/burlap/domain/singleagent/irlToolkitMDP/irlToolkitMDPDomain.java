package burlap.domain.singleagent.irlToolkitMDP;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import burlap.debugtools.RandomFactory;
import burlap.mdp.auxiliary.DomainGenerator;
import burlap.mdp.auxiliary.common.NullTermination;
import burlap.mdp.core.StateTransitionProb;
import burlap.mdp.core.TerminalFunction;
import burlap.mdp.core.action.Action;
import burlap.mdp.core.action.UniversalActionType;
import burlap.mdp.core.state.State;
import burlap.mdp.singleagent.common.VectorRF;
import burlap.mdp.singleagent.model.FactoredModel;
import burlap.mdp.singleagent.model.RewardFunction;
import burlap.mdp.singleagent.model.statemodel.FullStateModel;
import burlap.mdp.singleagent.oo.OOSADomain;
import burlap.domain.singleagent.irlToolkitMDP.state.irlToolkitMDPState;

public class irlToolkitMDPDomain implements DomainGenerator {
	
	public static final String ACTION_NORTH = "north";
	public static final String ACTION_SOUTH = "south";
	public static final String ACTION_EAST = "east";
	public static final String ACTION_WEST = "west";
	
	private final int width;
	private final int height;
	
	private int [][][] sa_s;
	private double [][][] sa_p;
	private double [] rewardVector;
	private double [][] features;
		
	public irlToolkitMDPDomain(int width, int height, int [][][] sa_s, double [][][] sa_p, double [] rewards, double [][] features) {
		this.width = width;
		this.height = height;
		this.setTransitionFunction(sa_s, sa_p);
		this.setRewardFunction(rewards);
		this.setFeatures(features);
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
	
	public double[] getRewardVector() {
		double[] copy = new double[rewardVector.length];
		for(int i = 0; i < rewardVector.length; i++) {
			copy[i] = rewardVector[i];
		}
		return copy;
	}
	
	@Override
	public OOSADomain generateDomain() {
		OOSADomain domain = new OOSADomain();
		domain.addStateClass("stateNumber", int.class);
		
		irlToolkitMDPModel smodel = new irlToolkitMDPModel(getAdjacencyList(), getTransitionProbs());
		RewardFunction rf = new VectorRF(getRewardVector());
		TerminalFunction tf = new NullTermination();
		
		FactoredModel model = new FactoredModel(smodel, rf, tf);
		domain.setModel(model);
		
		domain.addActionTypes(
				new UniversalActionType(ACTION_NORTH),
				new UniversalActionType(ACTION_SOUTH),
				new UniversalActionType(ACTION_EAST),
				new UniversalActionType(ACTION_WEST));
				
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
				irlToolkitMDPState ns = new irlToolkitMDPState(i);
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
			if(name.equals(ACTION_NORTH)){
				return 0;
			}
			else if(name.equals(ACTION_SOUTH)){
				return 1;
			}
			else if(name.equals(ACTION_EAST)){
				return 2;
			}
			else if(name.equals(ACTION_WEST)){
				return 3;
			}
			throw new RuntimeException("Unknown action " + name);
		}
	}
}
