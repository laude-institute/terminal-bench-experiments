--
-- PostgreSQL database dump
--

\ restrict 5iUEUwdh74RypxY2JwrbvgAH1ZGxeuVvvzVRyNxBySmG23YRg5gGMO2sYtUaFlG -- Dumped from database version 17.4
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
SET default_tablespace = '';
SET default_table_access_method = heap;
--
-- Name: agent; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.agent (
    name text NOT NULL,
    version text NOT NULL,
    created_at timestamp with time zone DEFAULT now() NOT NULL,
    description text
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
    CONSTRAINT job_check CHECK (
        (
            (git_commit_id IS NOT NULL)
            OR (package_version IS NOT NULL)
        )
    )
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
    cents_per_million_output_tokens integer
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
    path text NOT NULL
);
--
-- Name: trial; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.trial (
    id uuid NOT NULL,
    trial_name text NOT NULL,
    trial_uri text NOT NULL,
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
    created_at timestamp with time zone DEFAULT now() NOT NULL
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
    n_output_tokens integer
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
ADD CONSTRAINT dataset_task_pkey PRIMARY KEY (
        dataset_name,
        dataset_version,
        dataset_registry_uri,
        task_checksum
    );
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
ADD CONSTRAINT dataset_task_dataset_name_dataset_version_dataset_registry_fkey FOREIGN KEY (
        dataset_name,
        dataset_version,
        dataset_registry_uri
    ) REFERENCES public.dataset(name, version, registry_uri) ON DELETE CASCADE;
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

CREATE POLICY "Enable read access for all users" ON public.agent FOR
SELECT USING (true);
--
-- Name: dataset Enable read access for all users; Type: POLICY; Schema: public; Owner: -
--

CREATE POLICY "Enable read access for all users" ON public.dataset FOR
SELECT USING (true);
--
-- Name: dataset_task Enable read access for all users; Type: POLICY; Schema: public; Owner: -
--

CREATE POLICY "Enable read access for all users" ON public.dataset_task FOR
SELECT USING (true);
--
-- Name: job Enable read access for all users; Type: POLICY; Schema: public; Owner: -
--

CREATE POLICY "Enable read access for all users" ON public.job FOR
SELECT USING (true);
--
-- Name: model Enable read access for all users; Type: POLICY; Schema: public; Owner: -
--

CREATE POLICY "Enable read access for all users" ON public.model FOR
SELECT USING (true);
--
-- Name: task Enable read access for all users; Type: POLICY; Schema: public; Owner: -
--

CREATE POLICY "Enable read access for all users" ON public.task FOR
SELECT USING (true);
--
-- Name: trial Enable read access for all users; Type: POLICY; Schema: public; Owner: -
--

CREATE POLICY "Enable read access for all users" ON public.trial FOR
SELECT USING (true);
--
-- Name: trial_model Enable read access for all users; Type: POLICY; Schema: public; Owner: -
--

CREATE POLICY "Enable read access for all users" ON public.trial_model FOR
SELECT USING (true);
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

\ unrestrict 5iUEUwdh74RypxY2JwrbvgAH1ZGxeuVvvzVRyNxBySmG23YRg5gGMO2sYtUaFlG