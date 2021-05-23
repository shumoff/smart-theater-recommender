import sys
from recommender.api.server import serve
from recommender.core.migrations.migrate import apply, rollback

command_line_args = sys.argv[1:]

if command_line_args[0] == 'serve':
    serve()
elif command_line_args[0] == 'migrate':
    if command_line_args[1] == 'up':
        apply()
    elif command_line_args[1] == 'down':
        rollback()
else:
    print('Unknown command.')
