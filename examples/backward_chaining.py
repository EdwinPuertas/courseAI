"""First-Order Logic — Backward Chaining with AIMA FolKB.

Course: Artificial Intelligence · Unit 3 — Knowledge, Reasoning and Planning
Institution: Universidad Tecnológica de Bolívar
Reference: Russell, S. & Norvig, P. (2022). AIMA (4th ed.), Ch. 9 — Inference in FOL

Demonstrates backward chaining over a FOL knowledge base encoding the
"criminal law" example from R&N: West is an arms dealer selling missiles
to hostile nations, making him a criminal.
"""

import aima.utils
import aima.logic


def build_knowledge_base() -> aima.logic.FolKB:
    """Build and return a FOL knowledge base with the criminal law domain.

    Returns:
        FolKB populated with Horn-clause rules and ground facts encoding
        the arms-dealing scenario from R&N §9.3.
    """
    clauses = [
        # Rule: an American who sells weapons to a hostile nation is criminal
        aima.utils.expr("(American(x) & Weapon(y) & Sells(x, y, z) & Hostile(z)) ==> Criminal(x)"),
        # Facts about Nono
        aima.utils.expr("Enemy(Nono, America)"),
        aima.utils.expr("Owns(Nono, M1)"),
        aima.utils.expr("Missile(M1)"),
        # Rule: West sells any missile owned by Nono to Nono
        aima.utils.expr("(Missile(x) & Owns(Nono, x)) ==> Sells(West, x, Nono)"),
        # Facts about West
        aima.utils.expr("American(West)"),
        # Rule: missiles are weapons
        aima.utils.expr("Missile(x) ==> Weapon(x)"),
    ]
    kb = aima.logic.FolKB(clauses)

    # Additional enemy states (added dynamically)
    kb.tell(aima.utils.expr('Enemy(Coco, America)'))
    kb.tell(aima.utils.expr('Enemy(Jojo, America)'))
    kb.tell(aima.utils.expr("Enemy(x, America) ==> Hostile(x)"))

    return kb


def main() -> None:
    """Query the knowledge base for hostile entities and criminals."""
    kb = build_knowledge_base()

    # Backward-chaining queries
    hostile = kb.ask(aima.utils.expr('Hostile(x)'))
    criminal = kb.ask(aima.utils.expr('Criminal(x)'))

    print('Who is hostile?')
    print(hostile)

    print('\nWho is a criminal?')
    print(criminal)


if __name__ == "__main__":
    main()