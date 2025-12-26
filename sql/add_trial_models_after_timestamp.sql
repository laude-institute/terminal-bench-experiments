-- Add gemini trial model for all trials in the specific job.
-- Job: f62ad0bd-f897-425d-b47a-c1fee4d35326
-- Model to add:
--   ('google', 'gemini-3-pro-preview')
INSERT INTO public.trial_model (trial_id, model_name, model_provider)
SELECT t.id AS trial_id,
    model_data.model_name AS model_name,
    model_data.model_provider AS model_provider
FROM public.trial t
    CROSS JOIN (
        VALUES ('gemini-3-pro-preview', 'google')
    ) AS model_data(model_name, model_provider)
WHERE t.job_id = 'f62ad0bd-f897-425d-b47a-c1fee4d35326'::uuid
    AND NOT EXISTS (
        SELECT 1
        FROM public.trial_model tm
        WHERE tm.trial_id = t.id
            AND tm.model_name = model_data.model_name
            AND tm.model_provider = model_data.model_provider
    ) ON CONFLICT (trial_id, model_name, model_provider) DO NOTHING;