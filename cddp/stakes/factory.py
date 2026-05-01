"""
Stakes Factory — Centralized construction of stakes functions.
"""
from cddp.stakes.analytical import AnalyticalStakesDmin, AnalyticalStakesLsafe, AnalyticalFeasibilityTrigger, AnalyticalOracleStakes
from cddp.stakes.prospective import ProspectiveStakes, WrongProspectiveStakes

def build_stakes(config: dict, critics=None, oracle=None, actor=None):
    """
    Constructs the correct stakes function defined by the config.
    
    config['stakes'] schema:
    {
      'type': 'analytical' | 'prospective' | 'wrong_prospective',
      'analytic_mode': 'dmin' | 'lsafe' | 'feas',
      'wrong_mode': 'argmax' | 'pi_expectation',
      'p_ref': 1.0,
      'c_cat': -0.5,
      'tau': 0.5
    }
    """
    if config is None or 'stakes' not in config:
        return AnalyticalStakesDmin()
        
    st = config['stakes']
    stype = st.get('type', 'analytical')
    
    if stype == 'analytical':
        mode = st.get('analytic_mode', 'dmin')
        if mode == 'dmin':
            return AnalyticalStakesDmin()
        elif mode == 'lsafe':
            return AnalyticalStakesLsafe()
        elif mode == 'feas':
            return AnalyticalFeasibilityTrigger()
        elif mode == 'oracle':
            return AnalyticalOracleStakes()
        else:
            raise ValueError(f"Unknown analytic_mode: {mode}")
            
    elif stype == 'prospective':
        assert critics is not None and oracle is not None, "Prospective stakes require critics and oracle."
        return ProspectiveStakes(
            critics=critics,
            oracle=oracle,
            p_ref=st.get('p_ref', 1.0),
            c_cat=st.get('c_cat', -0.5)
        )
        
    elif stype == 'wrong_prospective':
        assert critics is not None and actor is not None, "WrongProspective stakes require critics and actor."
        return WrongProspectiveStakes(
            critics=critics,
            actor=actor,
            p_ref=st.get('p_ref', 1.0),
            c_cat=st.get('c_cat', -0.5),
            mode=st.get('wrong_mode', 'pi_expectation')
        )
        
    else:
        raise ValueError(f"Unknown stakes type: {stype}")
