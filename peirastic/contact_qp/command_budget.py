"""Spendable budget for frozen logical W/V epochs, never measured robot work.

An epoch holds both six-axis tool-frame quantities from a successful dual-device
publication until replacement, stop, or its review-time expiry. Expiry terminates
this mathematical output only: actual device stopping/tail work is unknown.
Measured-port monitoring has no reference to this balance.

An explicitly selected nominal-command source can fund final model output up to
the frozen baseline's outward power. Unused task authorization is discarded;
this supplied model makes no external one-port passivity claim.
"""
from dataclasses import dataclass, replace
import math
import numpy as np

from .port_constraint import PortEnergyConstraint
from .types import positive, vector


def _time(value):
    if isinstance(value,(bool,np.bool_)):raise ValueError('numeric logical time required')
    return positive(value,'logical time',zero=True)


@dataclass(frozen=True)
class _Pair:
    command_id: object
    wrench: np.ndarray
    velocity: np.ndarray
    rotation_base_tcp: np.ndarray
    reviewed_s: float
    expires_s: float
    power_w: float
    nominal_twist_tool: np.ndarray | None
    task_power_w: float
    committed_s: float | None = None


class CommandBudget:
    settlement_port='logical_final_model'
    wrench_convention='negative_control_raw_tcp_v1'

    def __init__(self,initial_j,capacity_j,stopping_reserve_j,*,max_command_interval_s,
                 settlement_port,wrench_convention,constraint=None,task_power_source='none'):
        if settlement_port!=self.settlement_port or wrench_convention!=self.wrench_convention:
            raise ValueError('explicit logical_final_model / negative_control_raw_tcp_v1 required')
        for value in (initial_j,capacity_j,stopping_reserve_j,max_command_interval_s):
            if isinstance(value,(bool,np.bool_)):raise ValueError('numeric logical budget fields required')
        self.balance_j=positive(initial_j,'initial_j',zero=True)
        self.capacity_j=positive(capacity_j,'capacity_j')
        self.stopping_reserve_j=positive(stopping_reserve_j,'stopping_reserve_j',zero=True)
        if not self.stopping_reserve_j<=self.balance_j<=self.capacity_j:
            raise ValueError('reserve <= initial <= capacity required')
        self.max_command_interval_s=positive(max_command_interval_s,'max_command_interval_s')
        if task_power_source not in ('none','nominal_command'):
            raise ValueError('explicit none or nominal_command task_power_source required')
        self.task_power_source=task_power_source
        self.assurance='two_port_command_model' if task_power_source=='nominal_command' else 'command_model'
        self.parameters=dict(constraint or {})
        if 'task_power_w' in self.parameters:
            raise ValueError('task power is derived only from the frozen nominal command')
        if self.parameters.get('assurance','command_model') not in ('command_model',self.assurance):
            raise ValueError('logical model assurance required')
        self.parameters['assurance']=self.assurance
        check=PortEnergyConstraint(wrench_environment=np.zeros(6),available_j=self.available_j,
            hold_s=self.max_command_interval_s,**self.parameters)
        if any(np.any(getattr(check,k)) for k in
                ('damping','tracking_error','wrench_error','wrench_rate')):
            raise ValueError('logical held model requires zero damping/error/rate terms')
        self.active=None;self.pending=None;self.started=False;self.latched_reason=None
        self.last_time_s=None;self._snapshot=None;self._snapshot_rotation=None
        self._snapshot_nominal=None
        self.cumulative_port_work_j=0.;self.cumulative_task_source_used_j=0.
        self.cumulative_tank_work_j=0.;self.cumulative_capacity_discard_j=0.
        self.events=[];self._seen=set()

    @property
    def reserved_j(self):
        pending=getattr(self,'pending',None);active=getattr(self,'active',None)
        new=0. if pending is None else max(0.,-pending.power_w-pending.task_power_w)*self.max_command_interval_s
        old=0. if active is None else max(0.,-active.power_w-active.task_power_w)*max(0.,active.expires_s-self.last_time_s)
        return old+new

    @property
    def available_j(self):
        return max(0.,self.raw_available_j)

    @property
    def raw_available_j(self):
        return self.balance_j-self.stopping_reserve_j-self.reserved_j

    @property
    def facts(self):
        return dict(balance_j=self.balance_j,reserved_j=self.reserved_j,available_j=self.available_j,
            raw_available_j=self.raw_available_j,command_budget_enforced=True,energy_constraint_enabled=True,
            settlement_port=self.settlement_port,wrench_convention=self.wrench_convention,
            max_command_interval_s=self.max_command_interval_s,latched_reason=self.latched_reason,
            logical_epoch_expiry_s=None if self.active is None else self.active.expires_s,
            task_power_source=self.task_power_source,energy_assurance=self.assurance,
            task_source_available_w=0. if self.active is None else self.active.task_power_w,
            cumulative_port_work_j=self.cumulative_port_work_j,
            cumulative_task_source_used_j=self.cumulative_task_source_used_j,
            cumulative_tank_work_j=self.cumulative_tank_work_j,
            cumulative_capacity_discard_j=self.cumulative_capacity_discard_j,
            physical_certified=False,actual_tail='unknown',predicted_recovery_credited_j=0.)

    @staticmethod
    def wrench_from_control(wrench_control_raw):
        return -vector(wrench_control_raw,(6,),name='six-axis control raw wrench')

    def drain_events(self):
        events=self.events;self.events=[]
        return events

    def _fault(self,reason):
        self.latched_reason=self.latched_reason or reason
        self.events.append(dict(event='logical_budget_fault',reason=reason,**self.facts))

    def fail(self,reason,*,now_s):
        try:self._settle(now_s)
        finally:self._fault(reason)

    def _settle(self,now_s,*,strict=False):
        try:now=_time(now_s)
        except (TypeError,ValueError,OverflowError):
            self._fault('nonfinite_or_invalid_logical_time')
            raise
        if self.last_time_s is not None and (now<self.last_time_s or (strict and now==self.last_time_s)):
            self._fault('repeated_or_reversed_logical_time')
            raise ValueError('repeated or reversed logical settlement time')
        old=self.active
        if old is not None:
            end=min(now,old.expires_s)
            dt=max(0.,end-self.last_time_s)
            source_used_w=min(old.task_power_w,max(0.,-old.power_w))
            port_work=dt*old.power_w
            source_work=dt*source_used_w
            change=port_work+source_work
            balance=self.balance_j+change
            discard=max(0.,balance-self.capacity_j)
            totals=(self.cumulative_port_work_j+port_work,
                self.cumulative_task_source_used_j+source_work,
                self.cumulative_tank_work_j+change,self.cumulative_capacity_discard_j+discard)
            if not all(math.isfinite(x) for x in (port_work,source_work,change,balance,discard,*totals)):
                self.latched_reason='nonfinite_logical_work'
                raise ValueError(self.latched_reason)
            # Upper clipping discards recovered energy. No lower clipping can
            # invent energy; an accounting violation fences further commands.
            self.balance_j=min(self.capacity_j,balance)
            (self.cumulative_port_work_j,self.cumulative_task_source_used_j,
             self.cumulative_tank_work_j,self.cumulative_capacity_discard_j)=totals
            self.events.append(dict(event='logical_epoch_work',command_id=old.command_id,
                start_s=self.last_time_s,end_s=end,power_w=old.power_w,work_j=port_work,
                port_work_j=port_work,task_source_available_w=old.task_power_w,
                task_source_used_w=source_used_w,task_source_used_j=source_work,
                tank_work_j=change,capacity_discard_j=discard,
                task_power_source=self.task_power_source,energy_assurance=self.assurance,
                power_sign='positive_environment_input',
                balance_j=self.balance_j,physical_certified=False))
            if self.balance_j<self.stopping_reserve_j-1e-12:
                self.latched_reason='logical_budget_underflow'
            if now>=old.expires_s:
                self.active=None
                self.latched_reason=self.latched_reason or 'logical_epoch_expired_actual_tail_unknown'
        self.last_time_s=now
        if self.raw_available_j < -1e-12:
            self._fault('logical_liability_exceeds_balance')
        return self.latched_reason is None

    def advance(self,now_s):
        """Strict external settlement tick; duplicate times cannot credit work."""
        return self._settle(now_s,strict=True)

    def snapshot(self,*,now_s,wrench_control_raw,rotation_base_tcp,nominal_twist_tool=None):
        if not self._settle(now_s):raise ValueError(self.latched_reason)
        if self.pending is not None:raise ValueError('unresolved logical reservation')
        rotation=vector(rotation_base_tcp,(3,3),name='pair rotation snapshot')
        if not np.allclose(rotation.T@rotation,np.eye(3),atol=1e-9) or np.linalg.det(rotation)<0:
            raise ValueError('invalid pair rotation snapshot')
        wrench=self.wrench_from_control(wrench_control_raw)
        nominal=None;task_power=0.
        if self.task_power_source=='nominal_command':
            nominal=vector(nominal_twist_tool,(6,),name='frozen nominal tool twist')
            with np.errstate(over='ignore',invalid='ignore'):
                nominal_power=float(wrench @ nominal)
            if not math.isfinite(nominal_power):
                raise ValueError('nonfinite nominal task power')
            task_power=max(0.,-nominal_power)
        elif nominal_twist_tool is not None:
            raise ValueError('nominal twist requires explicit nominal_command source')
        self._snapshot=PortEnergyConstraint(wrench_environment=wrench,
            available_j=self.available_j,hold_s=self.max_command_interval_s,
            task_power_w=task_power,**self.parameters)
        self._snapshot_rotation=rotation
        self._snapshot_nominal=nominal
        return self._snapshot

    def reserve(self,command_id,snapshot,final_velocity,*,now_s):
        if not self._settle(now_s):return False
        if snapshot is None or snapshot is not self._snapshot or self.pending is not None or command_id in self._seen:
            return False
        self._snapshot=None;self._seen.add(command_id)
        final=vector(final_velocity,(6,),name='final logical velocity')
        fresh=replace(snapshot,available_j=self.available_j)
        if not fresh.admissible(final,tolerance_w=0.,velocity_tolerance=0.):return False
        power=fresh.lower_power_w(final)
        expiry=float(now_s)+self.max_command_interval_s
        liability=max(0.,-power-fresh.task_power_w)*self.max_command_interval_s
        if not all(math.isfinite(x) for x in (power,expiry,liability)) or expiry<=now_s:
            raise ValueError('nonfinite logical reservation')
        if liability>self.available_j:return False
        self.pending=_Pair(command_id,fresh.wrench_environment,final,self._snapshot_rotation,
                           float(now_s),expiry,power,self._snapshot_nominal,fresh.task_power_w)
        self.started=False
        self.events.append(dict(event='logical_reservation',command_id=command_id,reviewed_s=now_s,
            expires_s=expiry,power_w=power,liability_j=liability,**self.facts))
        return True

    def publication_started(self,command_id):
        if self.pending is None or self.pending.command_id!=command_id or self.started:
            raise ValueError('unknown or repeated logical publication start')
        self.started=True
        self.events.append(dict(event='logical_publication_started',command_id=command_id,**self.facts))

    def commit(self,command_id,final_velocity,*,now_s,rotation_base_tcp,dual_success):
        if not self._settle(now_s):raise ValueError(self.latched_reason)
        pair=self.pending
        try:
            valid=(pair is not None and pair.command_id==command_id and self.started and dual_success is True
                and np.array_equal(vector(final_velocity,(6,)),pair.velocity)
                and np.array_equal(vector(rotation_base_tcp,(3,3)),pair.rotation_base_tcp)
                and now_s<pair.expires_s)
        except (TypeError,ValueError,OverflowError):valid=False
        if not valid:
            self.latched_reason='logical_commit_unproven_or_payload_changed'
            raise ValueError(self.latched_reason)
        # A new successful pair ends only the old LOGICAL output. Review delay
        # neither earns recovery nor extends the new pair's expiry.
        self.active=replace(pair,committed_s=float(now_s))
        self.pending=None;self.started=False
        self.events.append(dict(event='logical_epoch_commit',command_id=command_id,
            reviewed_s=pair.reviewed_s,committed_s=now_s,expires_s=pair.expires_s,
            wrench_tool=pair.wrench.tolist(),velocity_tool=pair.velocity.tolist(),
            nominal_twist_tool=None if pair.nominal_twist_tool is None else pair.nominal_twist_tool.tolist(),
            task_source_available_w=pair.task_power_w,task_power_source=self.task_power_source,
            energy_assurance=self.assurance,
            rotation_base_tcp=pair.rotation_base_tcp.tolist(),physical_certified=False))

    def reject_new_only(self,command_id,*,definitely_not_sent,now_s):
        self._settle(now_s)
        if type(definitely_not_sent) is not bool:raise ValueError('explicit no-send fact required')
        if self.pending is None or self.pending.command_id!=command_id:return False
        if definitely_not_sent and not self.started:
            self.pending=None
            self.events.append(dict(event='logical_candidate_not_sent',command_id=command_id,**self.facts))
            return True
        self.latched_reason=self.latched_reason or 'publication_partial_or_unknown'
        self.events.append(dict(event='logical_candidate_unknown',command_id=command_id,**self.facts))
        return False

    def stop(self,*,now_s):
        self._settle(now_s)
        self.active=None
        self.latched_reason=self.latched_reason or 'logical_output_stopped_actual_tail_unknown'
        self.events.append(dict(event='logical_output_stop',**self.facts))
