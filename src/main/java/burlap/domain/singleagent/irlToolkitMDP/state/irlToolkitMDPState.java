package burlap.domain.singleagent.irlToolkitMDP.state;

import java.util.List;
import java.util.Arrays;

import burlap.mdp.core.oo.state.MutableOOState;
import burlap.mdp.core.oo.state.ObjectInstance;
import burlap.mdp.core.state.MutableState;
import burlap.mdp.core.state.State;
import burlap.mdp.core.state.annotations.ShallowCopyState;

@ShallowCopyState
public class irlToolkitMDPState implements State {
	public int stateNumber;
	
	public irlToolkitMDPState(int stateNumber) {
		this.stateNumber = stateNumber;
	}

	@Override
	public List<Object> variableKeys() {
		List<Object> variableKeys = Arrays.<Object>asList("stateNumber");
		return variableKeys;
	}

	@Override
	public Object get(Object variableKey) {
		return this.stateNumber;
	}

	@Override
	public State copy() {
		return new irlToolkitMDPState(this.stateNumber);
	}

	
}
