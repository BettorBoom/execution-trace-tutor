-- Supabase SQL Editor에서 한 번 실행한다. API 키는 서버용 Secret에서만 사용한다.
create table if not exists public.user_settings (
    owner_id text primary key,
    key_ciphertext text,
    updated_at timestamptz not null default now()
);

create table if not exists public.tutorials (
    id uuid primary key,
    owner_id text not null references public.user_settings(owner_id),
    problem text not null,
    language text not null,
    source text not null,
    model text not null,
    quiz jsonb not null,
    progress jsonb not null,
    version integer not null default 0,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now(),
    completed_at timestamptz
);

create index if not exists tutorials_owner_updated_idx
    on public.tutorials (owner_id, updated_at desc);

alter table public.user_settings enable row level security;
alter table public.tutorials enable row level security;
revoke all on public.user_settings, public.tutorials from anon, authenticated;
grant select, insert, update, delete on public.user_settings, public.tutorials to service_role;
