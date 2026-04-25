import asyncio
from episteward import EpiStewardEnv, EpiAction

async def demo_all():
    env = EpiStewardEnv.in_process()

    tasks = {
        'task1_triage':     ('ceftriaxone',              'Single patient, learn correct prescribing'),
        'task2_containment':('piperacillin-tazobactam',  'ESBL cluster, contain spread'),
        'task3_outbreak':   ('colistin',                 'CRK network, allocate last-resort therapy'),
        'task4_multiagent': ('ceftriaxone',              'Multi-ward game, shift prescribing equilibrium'),
    }

    for task_id, (antibiotic, description) in tasks.items():
        print(f'=== {task_id.upper()} ===')
        print(f'Goal: {description}')
        result = await env.reset(task_id=task_id)
        obs = result.observation
        print(f'Patient: {obs.patient_id} | Ward: {obs.ward_id}')
        print(f'Infection: {obs.infection_site}')
        print(f'Resistance flags: {obs.resistance_flags}')

        action = EpiAction(
            antibiotic=antibiotic,
            dose_mg=1000,
            frequency_hours=8,
            duration_days=7,
            route='IV',
            culture_requested=True,
            target_app='pharmacy',
        )

        result = await env.step(action)
        print(f'Action: {antibiotic}')
        print(f'Reward: {result.reward:.3f}')
        print(f'Oversight flags: {result.info.get("oversight", {}).get("flags", [])}')
        print(f'Specialist stance: {result.info.get("specialist_feedback", {}).get("stance", "")}')
        print()

asyncio.run(demo_all())
