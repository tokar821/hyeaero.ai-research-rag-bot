-- Application users (sign up / sign in) and RBAC: user, admin, super_admin (at most one).

CREATE TABLE IF NOT EXISTS app_users (
    id              BIGSERIAL PRIMARY KEY,
    email           VARCHAR(320) NOT NULL,
    full_name       VARCHAR(255) NOT NULL,
    country         VARCHAR(100) NOT NULL,
    password_hash   VARCHAR(255) NOT NULL,
    role            VARCHAR(20) NOT NULL DEFAULT 'user',
    status          VARCHAR(20) NOT NULL DEFAULT 'pending',
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT app_users_role_check CHECK (role IN ('user', 'admin', 'super_admin')),
    CONSTRAINT app_users_status_check CHECK (status IN ('pending', 'active', 'rejected'))
);

CREATE UNIQUE INDEX IF NOT EXISTS app_users_email_lower_uidx ON app_users (lower(email));

CREATE UNIQUE INDEX IF NOT EXISTS app_users_one_super_admin ON app_users ((1)) WHERE role = 'super_admin';

CREATE INDEX IF NOT EXISTS app_users_role_status_idx ON app_users (role, status);
CREATE INDEX IF NOT EXISTS app_users_created_idx ON app_users (created_at DESC);

COMMENT ON TABLE app_users IS 'HyeAero.AI dashboard accounts; super_admin limited to one row via partial unique index';
