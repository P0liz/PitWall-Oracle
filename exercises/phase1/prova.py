import fastf1
session = fastf1.get_session(2026, 7, 'Q')
session.load()
print(session.results)