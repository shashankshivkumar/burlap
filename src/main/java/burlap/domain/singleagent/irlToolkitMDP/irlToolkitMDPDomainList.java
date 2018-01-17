package burlap.domain.singleagent.irlToolkitMDP;

import burlap.behavior.functionapproximation.dense.FromArrayFeatures;
import burlap.behavior.functionapproximation.dense.FromArraySAFeatures;
import burlap.behavior.policy.BoltzmannQPolicy;
import burlap.behavior.singleagent.Episode;
import burlap.behavior.singleagent.learnfromdemo.mlirl.commonrfs.LinearStateDifferentiableRF;
import burlap.behavior.singleagent.learnfromdemo.mlirl.differentiableplanners.DifferentiableParametricQLearning;
import burlap.behavior.singleagent.learnfromdemo.mlirl.differentiableplanners.DifferentiableVI;
import burlap.behavior.singleagent.learning.tdmethods.QLearning;
import burlap.domain.singleagent.irlToolkitMDP.irlToolkitMDPDomain.irlToolkitMDPModel;
import burlap.domain.singleagent.irlToolkitMDP.state.irlToolkitMDPListState;
import burlap.domain.singleagent.irlToolkitMDP.state.irlToolkitMDPState;
import burlap.mdp.auxiliary.DomainGenerator;
import burlap.mdp.auxiliary.common.NullTermination;
import burlap.mdp.core.Domain;
import burlap.mdp.core.StateTransitionProb;
import burlap.mdp.core.TerminalFunction;
import burlap.mdp.core.action.Action;
import burlap.mdp.core.action.UniversalActionType;
import burlap.mdp.core.state.State;
import burlap.mdp.singleagent.common.MatrixRF;
import burlap.mdp.singleagent.environment.Environment;
import burlap.mdp.singleagent.environment.SimulatedEnvironment;
import burlap.mdp.singleagent.model.FactoredModel;
import burlap.mdp.singleagent.model.RewardFunction;
import burlap.mdp.singleagent.model.statemodel.FullStateModel;
import burlap.mdp.singleagent.oo.OOSADomain;
import burlap.statehashing.simple.SimpleHashableStateFactory;

import java.util.ArrayList;
import java.util.List;

public class irlToolkitMDPDomainList implements DomainGenerator{
	
	private List<irlToolkitMDPDomain> mdpList;
	private double[][] features;
	private int numStatesPerMdp;
	
	public irlToolkitMDPDomainList(List<irlToolkitMDPDomain> mdps) {
		this.mdpList = mdps;
		this.setNumStates();
		this.setFeatures();
	}
	
	public int getNumMDPs() {
		return mdpList.size();
	}
	
	public irlToolkitMDPDomain getMDP(int mdpNumber) {
		return mdpList.get(mdpNumber);
	}
	
	public void setNumStates() {
		this.numStatesPerMdp = this.getMDP(0).getNumStates();
	}
	
	public void setFeatures() {
		int numMdps = mdpList.size();
		this.features = new double[numMdps * numStatesPerMdp][];
		
		for(int i = 0; i < numMdps; i++) {
			irlToolkitMDPDomain mdp = this.getMDP(i);
			double[][] subFeatures = mdp.getFeatures();
			for(int j = 0; j < numStatesPerMdp; j++) {
				this.features[i * numStatesPerMdp + j] = subFeatures[j];
			}
		}
	}
	
	@Override
	public OOSADomain generateDomain() {
		OOSADomain domain = new OOSADomain();
		domain.addStateClass("stateNumber", int.class).addStateClass("mdpNumber", int.class);//.addStateClass("features", double.class);
		
		irlToolkitMDPListModel smodel = new irlToolkitMDPListModel(mdpList);
		
		RewardFunction rf = new MatrixRF(mdpList);
		TerminalFunction tf = new NullTermination();
		
		FactoredModel model = new FactoredModel(smodel, rf, tf);
		domain.setModel(model);
		
		for(int i = 0; i < mdpList.get(0).getNumActions(); i++) {
			domain.addActionType(new UniversalActionType(Integer.toString(i)));
		}
		
		domain.addFeatureVector(features, numStatesPerMdp);
		
		return domain;
	}
	
	public static class irlToolkitMDPListModel implements FullStateModel {
		List<irlToolkitMDPModel> mdpModels = new ArrayList<irlToolkitMDPModel>();
		
		public irlToolkitMDPListModel(List<irlToolkitMDPDomain> mdps) {
			for(irlToolkitMDPDomain mdp : mdps) {
				this.mdpModels.add(new irlToolkitMDPModel(mdp.getAdjacencyList(), mdp.getTransitionProbs()));
			}
		}
		
		
		@Override
		public State sample(State s, Action a) {
			irlToolkitMDPState sp = (irlToolkitMDPState) mdpModels.get(((irlToolkitMDPListState)s).mdpNumber).sample(s, a);
			return new irlToolkitMDPListState(sp.stateNumber, ((irlToolkitMDPListState)s).mdpNumber);
		}

		@Override
		public List<StateTransitionProb> stateTransitions(State s, Action a) {
			List<StateTransitionProb> mdp_transitions = mdpModels.get(((irlToolkitMDPListState)s).mdpNumber).stateTransitions(s, a);
			List<StateTransitionProb> transitions = new ArrayList<StateTransitionProb>();

			for (StateTransitionProb tp_tmp : mdp_transitions) {
				irlToolkitMDPState ns = new irlToolkitMDPListState(((irlToolkitMDPState)tp_tmp.s).stateNumber, ((irlToolkitMDPListState)s).mdpNumber);
				StateTransitionProb tp = new StateTransitionProb((State)ns, tp_tmp.p);
				transitions.add(tp);
			}
			return transitions;
		}
		
	}
	
	public static void main(String[] args) {
		int[][][] sa_s0 = {{{0},{1}},{{0},{2}},{{1},{2}}};
		double[][][] sa_p0 = {{{1},{1}},{{1},{1}},{{1},{1}}};
		double[] rewardVector0 = {0,0,10};
		double[][] features0 = {{0,0},{0,1},{1,0}};
		int[][][] sa_s1 = {{{1},{0}},{{2},{0}},{{2},{1}}};
		double[][][] sa_p1 = {{{1},{1}},{{1},{1}},{{1},{1}}};
		double[] rewardVector1 = {10,0,0};
		double[][] features1 = {{1,0},{0,1},{0,0}};
		
		List<irlToolkitMDPDomain> mdps = new ArrayList<irlToolkitMDPDomain>();
		mdps.add(new irlToolkitMDPDomain(sa_s0, sa_p0, rewardVector0, features0));
		mdps.add(new irlToolkitMDPDomain(sa_s1, sa_p1, rewardVector1, features1));
		irlToolkitMDPDomainList mdpList = new irlToolkitMDPDomainList(mdps);
		
		OOSADomain domain = mdpList.generateDomain();
		
		FromArrayFeatures featureGen = new FromArrayFeatures(domain);
		LinearStateDifferentiableRF rf = new LinearStateDifferentiableRF(featureGen, 2, false);
		
		DifferentiableVI planner = new DifferentiableVI(domain, rf, 0.99, 0.5, new SimpleHashableStateFactory(), 0.01, 500);
		
		planner.planFromState(new irlToolkitMDPListState(0, 0));
	}
}
