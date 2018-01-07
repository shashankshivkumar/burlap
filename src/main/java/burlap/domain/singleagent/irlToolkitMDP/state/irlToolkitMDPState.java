package burlap.domain.singleagent.irlToolkitMDP.state;

import java.util.List;

import burlap.mdp.core.oo.state.MutableOOState;
import burlap.mdp.core.oo.state.ObjectInstance;
import burlap.mdp.core.state.MutableState;
import burlap.mdp.core.state.State;
import burlap.mdp.core.state.annotations.ShallowCopyState;

@ShallowCopyState
public class irlToolkitMDPState implements MutableOOState {
	public int stateNumber;
	
	public irlToolkitMDPState(int stateNumber) {
		this.stateNumber = stateNumber;
	}

	@Override
	public int numObjects() {
		return 0;
	}

	@Override
	public ObjectInstance object(String oname) {
		return null;
	}

	@Override
	public List<ObjectInstance> objects() {
		return null;
	}

	@Override
	public List<ObjectInstance> objectsOfClass(String oclass) {
		return null;
	}

	@Override
	public List<Object> variableKeys() {
		return null;
	}

	@Override
	public Object get(Object variableKey) {
		return null;
	}

	@Override
	public State copy() {
		return new irlToolkitMDPState(stateNumber);
	}

	@Override
	public MutableState set(Object variableKey, Object value) {
		throw new RuntimeException("Cannot modify objects to irlToolkitMDPState.");
	}

	@Override
	public MutableOOState addObject(ObjectInstance o) {
		throw new RuntimeException("Cannot add objects to irlToolkitMDPState.");
	}

	@Override
	public MutableOOState removeObject(String oname) {
		throw new RuntimeException("Cannot remove objects from irlToolkitMDPState.");
	}

	@Override
	public MutableOOState renameObject(String objectName, String newName) {
		throw new RuntimeException("Cannot modify objects in irlToolkitMDPState.");
	}
}
