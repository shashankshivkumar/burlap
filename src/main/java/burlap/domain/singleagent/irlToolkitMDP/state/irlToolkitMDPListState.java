package burlap.domain.singleagent.irlToolkitMDP.state;

import java.util.List;
import java.util.Arrays;

import burlap.mdp.core.state.State;

public class irlToolkitMDPListState implements State {
	public int stateNumber;
	public int mdpNumber;
	
	public irlToolkitMDPListState(int stateNumber, int mdpNumber) {
		this.stateNumber = stateNumber;
		this.mdpNumber = mdpNumber;
	}
	
	@Override
	public List<Object> variableKeys() {
		List<Object> variableKeys = Arrays.<Object>asList("stateNumber", "mdpNumber");
		return variableKeys;
	}

	@Override
	public Object get(Object variableKey) {
		
		if(variableKey.equals("stateNumber")) {
			return stateNumber;
		}
		return mdpNumber;
	}

	@Override
	public State copy() {
		return new irlToolkitMDPListState(this.stateNumber, this.mdpNumber);
	}

}
