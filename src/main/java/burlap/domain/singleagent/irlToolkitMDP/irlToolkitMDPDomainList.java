package burlap.domain.singleagent.irlToolkitMDP;

import burlap.domain.singleagent.irlToolkitMDP.irlToolkitMDPDomain.irlToolkitMDPModel;
import burlap.domain.singleagent.irlToolkitMDP.state.irlToolkitMDPListState;
import burlap.mdp.auxiliary.DomainGenerator;
import burlap.mdp.auxiliary.common.NullTermination;
import burlap.mdp.core.Domain;
import burlap.mdp.core.StateTransitionProb;
import burlap.mdp.core.TerminalFunction;
import burlap.mdp.core.action.Action;
import burlap.mdp.core.action.UniversalActionType;
import burlap.mdp.core.state.State;
import burlap.mdp.singleagent.common.MatrixRF;
import burlap.mdp.singleagent.model.FactoredModel;
import burlap.mdp.singleagent.model.RewardFunction;
import burlap.mdp.singleagent.model.statemodel.FullStateModel;
import burlap.mdp.singleagent.oo.OOSADomain;

import java.util.List;

public class irlToolkitMDPDomainList implements DomainGenerator{
	
	private List<irlToolkitMDPDomain> mdpList;
	
	public irlToolkitMDPDomainList(List<irlToolkitMDPDomain> mdps) {
		this.mdpList = mdps;
	}
	
	public int getNumMDPs() {
		return mdpList.size();
	}
	
	public irlToolkitMDPDomain getMDP(int mdpNumber) {
		return mdpList.get(mdpNumber);
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
		
		return domain;
	}
	
	public static class irlToolkitMDPListModel implements FullStateModel {
		List<irlToolkitMDPModel> mdpModels;
		
		public irlToolkitMDPListModel(List<irlToolkitMDPDomain> mdps) {
			for(irlToolkitMDPDomain mdp : mdps) {
				this.mdpModels.add(new irlToolkitMDPModel(mdp.getAdjacencyList(), mdp.getTransitionProbs()));
			}
		}
		
		
		@Override
		public State sample(State s, Action a) {
			return mdpModels.get(((irlToolkitMDPListState)s).mdpNumber).sample(s, a);
		}

		@Override
		public List<StateTransitionProb> stateTransitions(State s, Action a) {
			return mdpModels.get(((irlToolkitMDPListState)s).mdpNumber).stateTransitions(s, a);
		}
		
	}
}
