{
  description = "moeinsum: mixed Python and Mojo package scaffolded by mohaus";

  inputs = {
    git-hooks-nix.url = "github:cachix/git-hooks.nix";
    git-hooks-nix.inputs.nixpkgs.follows = "nixpkgs";
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    mohaus.url = "github:aarnphm/mohaus";
    mohaus.inputs.git-hooks-nix.follows = "git-hooks-nix";
    mohaus.inputs.nixpkgs.follows = "nixpkgs";
  };

  outputs = {
    self,
    git-hooks-nix,
    mohaus,
    nixpkgs,
    ...
  }: let
    systems = [
      "aarch64-darwin"
      "aarch64-linux"
      "x86_64-linux"
    ];

    forAllSystems = fn:
      nixpkgs.lib.genAttrs systems (
        system:
          fn system (
            import nixpkgs {
              inherit system;
            }
          )
      );

    devPackages = system: pkgs: [
      mohaus.packages.${system}.default
      pkgs.alejandra
      pkgs.coreutils
      pkgs.deadnix
      pkgs.python311
      pkgs.ruff
      pkgs.statix
      pkgs.uv
    ];

    mojoFormatHook = pkgs:
      pkgs.writeShellApplication {
        name = "moeinsum-mojo-format-check";
        runtimeInputs = [
          pkgs.coreutils
          pkgs.diffutils
        ];
        text = ''
          resolve_mojo() {
            if [ -n "''${MOHAUS_MOJO:-}" ] && [ -x "''${MOHAUS_MOJO:-}" ]; then
              printf '%s\n' "$MOHAUS_MOJO"
              return 0
            fi

            if command -v mojo >/dev/null 2>&1; then
              command -v mojo
              return 0
            fi

            if [ -n "''${MODULAR_DERIVED_PATH:-}" ] && [ -x "''${MODULAR_DERIVED_PATH:-}/build/bin/mojo" ]; then
              printf '%s\n' "$MODULAR_DERIVED_PATH/build/bin/mojo"
              return 0
            fi

            if [ -n "''${MODULAR_HOME:-}" ] && [ -x "''${MODULAR_HOME:-}/bin/mojo" ]; then
              printf '%s\n' "$MODULAR_HOME/bin/mojo"
              return 0
            fi

            return 1
          }

          if [ "$#" -eq 0 ]; then
            exit 0
          fi

          if ! mojo="$(resolve_mojo)"; then
            printf '%s\n' "mojo format skipped: no executable found via MOHAUS_MOJO, PATH, MODULAR_DERIVED_PATH, or MODULAR_HOME" >&2
            exit 0
          fi

          tmp="$(mktemp -d)"
          trap 'rm -rf "$tmp"' EXIT

          failed=0
          index=0
          for source in "$@"; do
            if [ ! -f "$source" ]; then
              continue
            fi

            case "$source" in
              *.mojo) ;;
              *) continue ;;
            esac

            copy="$tmp/$index.mojo"
            cp "$source" "$copy"
            "$mojo" format --line-length 119 --quiet "$copy"
            if ! diff -u "$source" "$copy" >&2; then
              printf '%s\n' "mojo format check failed: $source" >&2
              failed=1
            fi
            index=$((index + 1))
          done

          if [ "$failed" -ne 0 ]; then
            printf '%s\n' "run: mojo format --line-length 119 <files>" >&2
          fi

          exit "$failed"
        '';
      };

    mkCommandApp = system: pkgs: name: text: {
      type = "app";
      program = "${
        pkgs.writeShellApplication {
          inherit name text;
          runtimeInputs = devPackages system pkgs;
        }
      }/bin/${name}";
    };
  in {
    devShells = forAllSystems (
      system: pkgs: let
        preCommit = self.checks.${system}.pre-commit;
      in {
        default = pkgs.mkShell {
          packages = (devPackages system pkgs) ++ preCommit.enabledPackages;

          env = {
            UV_PYTHON = "${pkgs.python311}/bin/python";
          };

          shellHook = ''
            export VIRTUAL_ENV="$PWD/.venv"
            if [ ! -x "$VIRTUAL_ENV/bin/python" ]; then
              uv venv --python "${pkgs.python311}/bin/python" "$VIRTUAL_ENV"
            fi

            export PATH="$VIRTUAL_ENV/bin:$PATH"
            export UV_PYTHON="$VIRTUAL_ENV/bin/python"

            stamp="$VIRTUAL_ENV/.mohaus-editable-stamp"
            if [ ! -f "$stamp" ] \
              || [ pyproject.toml -nt "$stamp" ] \
              || [ .mojo-version -nt "$stamp" ] \
              || [ -n "$(find src python -type f -newer "$stamp" -print -quit 2>/dev/null)" ]; then
              mohaus develop
              touch "$stamp"
            fi

            if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
              ${preCommit.shellHook}
            fi
          '';
        };
      }
    );

    checks = forAllSystems (
      system: pkgs: {
        pre-commit = git-hooks-nix.lib.${system}.run {
          src = ./.;
          hooks = {
            mojo-format = {
              enable = true;
              name = "mojo format";
              entry = "${mojoFormatHook pkgs}/bin/moeinsum-mojo-format-check";
              files = "\\.mojo$";
            };

            ruff-format = {
              enable = true;
              name = "ruff format";
              entry = "${pkgs.ruff}/bin/ruff format --check";
              files = "\\.(py|pyi)$";
            };

            ruff-check = {
              enable = true;
              name = "ruff check";
              entry = "${pkgs.ruff}/bin/ruff check";
              files = "\\.(py|pyi)$";
            };

            ty = {
              enable = true;
              name = "ty check";
              entry = "${pkgs.uv}/bin/uvx ty check";
              pass_filenames = false;
              files = "\\.(py|pyi)$";
            };

            alejandra = {
              enable = true;
              name = "alejandra";
              entry = "${pkgs.alejandra}/bin/alejandra --check";
              files = "\\.nix$";
            };

            deadnix = {
              enable = true;
              name = "deadnix";
              entry = "${pkgs.deadnix}/bin/deadnix --fail flake.nix";
              pass_filenames = false;
              files = "\\.nix$";
            };

            statix = {
              enable = true;
              name = "statix";
              entry = "${pkgs.statix}/bin/statix check flake.nix";
              pass_filenames = false;
              files = "\\.nix$";
            };

            check-added-large-files.enable = true;
            check-json.enable = true;
            check-merge-conflicts.enable = true;
            check-toml.enable = true;
            check-yaml.enable = true;
            end-of-file-fixer.enable = true;
            trim-trailing-whitespace.enable = true;
          };
        };
      }
    );

    apps = forAllSystems (
      system: pkgs: let
        mohausApp = {
          type = "app";
          program = "${mohaus.packages.${system}.default}/bin/mohaus";
        };
      in {
        default = mohausApp;
        mohaus = mohausApp;

        develop = mkCommandApp system pkgs "moeinsum-develop" ''
          mohaus develop "$@"
        '';

        build = mkCommandApp system pkgs "moeinsum-build" ''
          mohaus build "$@"
        '';

        sdist = mkCommandApp system pkgs "moeinsum-sdist" ''
          mohaus sdist "$@"
        '';

        fmt = mkCommandApp system pkgs "moeinsum-fmt" ''
          ruff format python
          alejandra flake.nix
        '';

        check = mkCommandApp system pkgs "moeinsum-check" ''
          ruff format --check python
          ruff check python
          uvx ty check
          alejandra --check flake.nix
          deadnix --fail flake.nix
          statix check flake.nix
          out_dir="$(mktemp -d)"
          trap 'rm -rf "$out_dir"' EXIT
          mohaus sdist --out "$out_dir"
        '';
      }
    );

    formatter = forAllSystems (_system: pkgs: pkgs.alejandra);
  };
}
