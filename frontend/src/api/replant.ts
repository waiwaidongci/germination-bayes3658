import {request} from './client';import {ReplantPlan,ReplantLotInput} from '../types/domain';
export const getReplantPlans=()=>request<ReplantPlan[]>('/replant-plans');
export const getReplantPlan=(id:number)=>request<ReplantPlan>(`/replant-plans/${id}`);
export const createReplantPlan=(data:{name:string;run_id:number;target:number;required_probability:number;time_limit:number;lots:ReplantLotInput[]})=>request<ReplantPlan>('/replant-plans',{method:'POST',body:JSON.stringify(data)});
