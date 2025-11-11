--
-- PostgreSQL database dump
--

\restrict 402vamF8CJgTKwhKpbtkEtvtdYrGmCwfgb1JGbOwn7DfRaJ7uYawI1R750AL2uD

-- Dumped from database version 17.4
-- Dumped by pg_dump version 17.6 (Homebrew)

SET statement_timeout = 0;
SET lock_timeout = 0;
SET idle_in_transaction_session_timeout = 0;
SET transaction_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SELECT pg_catalog.set_config('search_path', '', false);
SET check_function_bodies = false;
SET xmloption = content;
SET client_min_messages = warning;
SET row_security = off;

--
-- Name: public; Type: SCHEMA; Schema: -; Owner: -
--

CREATE SCHEMA public;


--
-- Name: SCHEMA public; Type: COMMENT; Schema: -; Owner: -
--

COMMENT ON SCHEMA public IS 'standard public schema';


--
-- Name: get_agent_scores(text, text); Type: FUNCTION; Schema: public; Owner: -
--

CREATE FUNCTION public.get_agent_scores(p_dataset_name text, p_dataset_version text) RETURNS TABLE(rank bigint, agent_name text, agent_version text, model_name text, model_provider text, accuracy numeric, stderr numeric, agent_display_name text, model_display_name text, agent_org_display_name text, model_org_display_name text, agent_url text, created_at timestamp with time zone)
    LANGUAGE plpgsql
    AS $$ BEGIN RETURN QUERY WITH p_hats AS (
        SELECT t.agent_name,
            t.agent_version,
            tm.model_name,
            tm.model_provider,
            task.name AS task_name,
            AVG(COALESCE(t.reward, 0)) AS p_hat,
            COUNT(*) AS n_trials,
            MIN(t.created_at) AS earliest_trial_date,
            AVG(
                jsonb_array_length(t.agent_metadata->'api_request_times_msec')
            ) AS avg_api_calls,
            SUM(
                CASE
                    WHEN t.exception_info IS NULL THEN 0
                    ELSE 1
                END
            ) AS n_errors,
            CASE
                WHEN COUNT(*) > 1 THEN AVG(COALESCE(t.reward, 0)) * (1 - AVG(COALESCE(t.reward, 0))) / (COUNT(*) - 1)
                ELSE NULL
            END AS partial_var,
            AVG(tm.n_input_tokens) AS avg_n_input_tokens,
            AVG(tm.n_output_tokens) AS avg_n_output_tokens,
            AVG(
                tm.n_input_tokens / 1000000.0 * m.cents_per_million_input_tokens + tm.n_output_tokens / 1000000.0 * m.cents_per_million_output_tokens
            ) AS avg_cost_cents,
            AVG(
                EXTRACT(
                    EPOCH
                    FROM (
                            t.agent_execution_ended_at - t.agent_execution_started_at
                        )
                )
            ) AS avg_execution_time_seconds
        FROM trial AS t
            INNER JOIN dataset_task AS dt ON dt.task_checksum = t.task_checksum
            INNER JOIN task ON task.checksum = dt.task_checksum
            INNER JOIN trial_model AS tm ON tm.trial_id = t.id
            INNER JOIN model AS m ON m.name = tm.model_name
            AND m.provider = tm.model_provider
            INNER JOIN job AS j ON j.id = t.job_id
        WHERE dt.dataset_name = p_dataset_name
            AND dt.dataset_version = p_dataset_version
            AND (
                t.exception_info IS NULL
                OR t.exception_info->>'exception_type' IN (
                    'AgentTimeoutError',
                    'VerifierTimeoutError'
                )
            )
        GROUP BY t.agent_name,
            t.agent_version,
            tm.model_name,
            tm.model_provider,
            task.name
    ),
    aggregated_scores AS (
        SELECT ph.agent_name,
            ph.agent_version,
            ph.model_name,
            ph.model_provider,
            AVG(ph.p_hat) AS accuracy,
            CASE
                WHEN COUNT(*) > COUNT(ph.partial_var) THEN NULL
                ELSE SQRT(SUM(ph.partial_var)) / COUNT(*)
            END AS stderr,
            MIN(ph.earliest_trial_date) AS created_at
        FROM p_hats ph
        GROUP BY ph.agent_name,
            ph.agent_version,
            ph.model_name,
            ph.model_provider
        HAVING AVG(ph.p_hat) > 0.01
    )
SELECT RANK() OVER (
        ORDER BY a.accuracy DESC
    ) AS rank,
    a.agent_name,
    a.agent_version,
    a.model_name,
    a.model_provider,
    a.accuracy,
    a.stderr,
    ag.display_name AS agent_display_name,
    mo.display_name AS model_display_name,
    ag.org_display_name AS agent_org_display_name,
    mo.org_display_name AS model_org_display_name,
    ag.url AS agent_url,
    a.created_at
FROM aggregated_scores a
    LEFT JOIN agent ag ON ag.name = a.agent_name
    AND ag.version = a.agent_version
    LEFT JOIN model mo ON mo.name = a.model_name
    AND mo.provider = a.model_provider
ORDER BY a.accuracy DESC;
END;
$$;


SET default_tablespace = '';

SET default_table_access_method = heap;

--
-- Name: agent; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.agent (
    name text NOT NULL,
    version text NOT NULL,
    created_at timestamp with time zone DEFAULT now() NOT NULL,
    description text,
    org_display_name text,
    url text,
    display_name text
);


--
-- Name: dataset; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.dataset (
    name text NOT NULL,
    version text NOT NULL,
    registry_uri text NOT NULL,
    created_at timestamp with time zone DEFAULT now() NOT NULL,
    description text
);


--
-- Name: dataset_task; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.dataset_task (
    dataset_name text NOT NULL,
    dataset_version text NOT NULL,
    dataset_registry_uri text NOT NULL,
    task_checksum text NOT NULL,
    created_at timestamp with time zone DEFAULT now() NOT NULL
);


--
-- Name: job; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.job (
    id uuid NOT NULL,
    created_at timestamp with time zone DEFAULT now() NOT NULL,
    job_name text NOT NULL,
    username text NOT NULL,
    started_at timestamp with time zone,
    ended_at timestamp with time zone,
    git_commit_id text,
    package_version text,
    n_trials integer NOT NULL,
    config jsonb NOT NULL,
    metrics jsonb,
    stats jsonb,
    CONSTRAINT job_check CHECK (((git_commit_id IS NOT NULL) OR (package_version IS NOT NULL)))
);


--
-- Name: model; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.model (
    name text NOT NULL,
    provider text NOT NULL,
    created_at timestamp with time zone DEFAULT now() NOT NULL,
    description text,
    cents_per_million_input_tokens integer,
    cents_per_million_output_tokens integer,
    display_name text,
    org_display_name text
);


--
-- Name: task; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.task (
    checksum text NOT NULL,
    created_at timestamp with time zone DEFAULT now() NOT NULL,
    source text,
    name text NOT NULL,
    instruction text NOT NULL,
    agent_timeout_sec numeric NOT NULL,
    verifier_timeout_sec numeric NOT NULL,
    git_url text,
    git_commit_id text,
    path text NOT NULL,
    metadata jsonb
);


--
-- Name: trial; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.trial (
    id uuid NOT NULL,
    trial_name text NOT NULL,
    trial_uri text,
    job_id uuid,
    task_checksum text NOT NULL,
    agent_name text NOT NULL,
    agent_version text NOT NULL,
    reward numeric,
    started_at timestamp with time zone,
    ended_at timestamp with time zone,
    environment_setup_started_at timestamp with time zone,
    environment_setup_ended_at timestamp with time zone,
    agent_setup_started_at timestamp with time zone,
    agent_setup_ended_at timestamp with time zone,
    agent_execution_started_at timestamp with time zone,
    agent_execution_ended_at timestamp with time zone,
    verifier_started_at timestamp with time zone,
    verifier_ended_at timestamp with time zone,
    config jsonb NOT NULL,
    exception_info jsonb,
    created_at timestamp with time zone DEFAULT now() NOT NULL,
    agent_metadata jsonb
);


--
-- Name: trial_model; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.trial_model (
    trial_id uuid NOT NULL,
    created_at timestamp with time zone DEFAULT now() NOT NULL,
    model_name text NOT NULL,
    model_provider text NOT NULL,
    n_input_tokens integer,
    n_output_tokens integer,
    n_cache_tokens integer
);


--
-- Name: agent agent_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.agent
    ADD CONSTRAINT agent_pkey PRIMARY KEY (name, version);


--
-- Name: dataset dataset_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.dataset
    ADD CONSTRAINT dataset_pkey PRIMARY KEY (name, version, registry_uri);


--
-- Name: dataset_task dataset_task_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.dataset_task
    ADD CONSTRAINT dataset_task_pkey PRIMARY KEY (dataset_name, dataset_version, dataset_registry_uri, task_checksum);


--
-- Name: job job_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.job
    ADD CONSTRAINT job_pkey PRIMARY KEY (id);


--
-- Name: model model_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.model
    ADD CONSTRAINT model_pkey PRIMARY KEY (name, provider);


--
-- Name: task task_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.task
    ADD CONSTRAINT task_pkey PRIMARY KEY (checksum);


--
-- Name: task task_source_name_key; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.task
    ADD CONSTRAINT task_source_name_key UNIQUE (source, name);


--
-- Name: trial_model trial_model_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.trial_model
    ADD CONSTRAINT trial_model_pkey PRIMARY KEY (trial_id, model_name, model_provider);


--
-- Name: trial trial_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.trial
    ADD CONSTRAINT trial_pkey PRIMARY KEY (id);


--
-- Name: dataset_task dataset_task_dataset_name_dataset_version_dataset_registry_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.dataset_task
    ADD CONSTRAINT dataset_task_dataset_name_dataset_version_dataset_registry_fkey FOREIGN KEY (dataset_name, dataset_version, dataset_registry_uri) REFERENCES public.dataset(name, version, registry_uri) ON UPDATE CASCADE ON DELETE CASCADE;


--
-- Name: dataset_task dataset_task_task_checksum_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.dataset_task
    ADD CONSTRAINT dataset_task_task_checksum_fkey FOREIGN KEY (task_checksum) REFERENCES public.task(checksum) ON DELETE CASCADE;


--
-- Name: trial trial_agent_name_agent_version_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.trial
    ADD CONSTRAINT trial_agent_name_agent_version_fkey FOREIGN KEY (agent_name, agent_version) REFERENCES public.agent(name, version) ON DELETE CASCADE;


--
-- Name: trial trial_job_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.trial
    ADD CONSTRAINT trial_job_id_fkey FOREIGN KEY (job_id) REFERENCES public.job(id) ON DELETE CASCADE;


--
-- Name: trial_model trial_model_model_name_model_provider_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.trial_model
    ADD CONSTRAINT trial_model_model_name_model_provider_fkey FOREIGN KEY (model_name, model_provider) REFERENCES public.model(name, provider) ON DELETE CASCADE;


--
-- Name: trial_model trial_model_trial_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.trial_model
    ADD CONSTRAINT trial_model_trial_id_fkey FOREIGN KEY (trial_id) REFERENCES public.trial(id) ON DELETE CASCADE;


--
-- Name: trial trial_task_checksum_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.trial
    ADD CONSTRAINT trial_task_checksum_fkey FOREIGN KEY (task_checksum) REFERENCES public.task(checksum) ON DELETE CASCADE;


--
-- Name: agent Enable read access for all users; Type: POLICY; Schema: public; Owner: -
--

CREATE POLICY "Enable read access for all users" ON public.agent FOR SELECT USING (true);


--
-- Name: dataset Enable read access for all users; Type: POLICY; Schema: public; Owner: -
--

CREATE POLICY "Enable read access for all users" ON public.dataset FOR SELECT USING (true);


--
-- Name: dataset_task Enable read access for all users; Type: POLICY; Schema: public; Owner: -
--

CREATE POLICY "Enable read access for all users" ON public.dataset_task FOR SELECT USING (true);


--
-- Name: job Enable read access for all users; Type: POLICY; Schema: public; Owner: -
--

CREATE POLICY "Enable read access for all users" ON public.job FOR SELECT USING (true);


--
-- Name: model Enable read access for all users; Type: POLICY; Schema: public; Owner: -
--

CREATE POLICY "Enable read access for all users" ON public.model FOR SELECT USING (true);


--
-- Name: task Enable read access for all users; Type: POLICY; Schema: public; Owner: -
--

CREATE POLICY "Enable read access for all users" ON public.task FOR SELECT USING (true);


--
-- Name: trial Enable read access for all users; Type: POLICY; Schema: public; Owner: -
--

CREATE POLICY "Enable read access for all users" ON public.trial FOR SELECT USING (true);


--
-- Name: trial_model Enable read access for all users; Type: POLICY; Schema: public; Owner: -
--

CREATE POLICY "Enable read access for all users" ON public.trial_model FOR SELECT USING (true);


--
-- Name: agent; Type: ROW SECURITY; Schema: public; Owner: -
--

ALTER TABLE public.agent ENABLE ROW LEVEL SECURITY;

--
-- Name: dataset; Type: ROW SECURITY; Schema: public; Owner: -
--

ALTER TABLE public.dataset ENABLE ROW LEVEL SECURITY;

--
-- Name: dataset_task; Type: ROW SECURITY; Schema: public; Owner: -
--

ALTER TABLE public.dataset_task ENABLE ROW LEVEL SECURITY;

--
-- Name: job; Type: ROW SECURITY; Schema: public; Owner: -
--

ALTER TABLE public.job ENABLE ROW LEVEL SECURITY;

--
-- Name: model; Type: ROW SECURITY; Schema: public; Owner: -
--

ALTER TABLE public.model ENABLE ROW LEVEL SECURITY;

--
-- Name: task; Type: ROW SECURITY; Schema: public; Owner: -
--

ALTER TABLE public.task ENABLE ROW LEVEL SECURITY;

--
-- Name: trial; Type: ROW SECURITY; Schema: public; Owner: -
--

ALTER TABLE public.trial ENABLE ROW LEVEL SECURITY;

--
-- Name: trial_model; Type: ROW SECURITY; Schema: public; Owner: -
--

ALTER TABLE public.trial_model ENABLE ROW LEVEL SECURITY;

--
-- PostgreSQL database dump complete
--

\unrestrict 402vamF8CJgTKwhKpbtkEtvtdYrGmCwfgb1JGbOwn7DfRaJ7uYawI1R750AL2uD

