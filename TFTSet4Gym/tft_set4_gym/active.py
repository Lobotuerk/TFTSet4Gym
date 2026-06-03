from . import stats as stats
from . import field as field
from . import origin_class_stats as origin_class_stats
from .combat_state import get_state
import random


# changing the stat manually since we have shenanigans in place for AD change in the section that makes stat changes
def jhin_init(champion):
    # champion.add_que('change_stat', -30, None, 'AS', stats.ATTACKS_PER_SECOND_FIXED[champion.name][champion.stars])
    champion.AS = stats.ATTACKS_PER_SECOND_FIXED[champion.name][champion.stars]
    champion.print(' {} {} --> {}'.format('AS', None, champion.AS))


def jhin(champion, target):
    s = get_state()

    found = False
    index = -1
    if len(s.jhin_shots) > 0:
        for i, v in enumerate(s.jhin_shots):
            if(v[0] == champion):
                found = True
                index = i
                break

    if found:
        s.jhin_shots[index][1] += 1
    else:
        s.jhin_shots.append([champion, 1])
        index = len(s.jhin_shots) - 1

    if s.jhin_shots[index][1] == stats.ACTIVATE_EVERY_X_ATTACKS[champion.name]:
        s.jhin_shots[index][1] = 0
        champion.print(' active triggered')
        # the last 100% of the damage is added in 'champion_functions.py: attack()'
        return {'damage': (stats.ACTIVE_DMG_PERCENT[champion.name][champion.stars] * champion.SP - 1) * champion.AD, 'true_damage': False, 'crit_random': None, 'dodge_random': None}

    else: 
        return {'damage': 0, 'true_damage': False, 'crit_random': None, 'dodge_random': None}


def kalista(champion, target):
    s = get_state()
    if target:
        found = False
        index = -1
        if len(s.kalista_targets) > 0:
            for i, v in enumerate(s.kalista_targets):
                if(v[0] == champion, v[1] == target):
                    found = True
                    index = i
                    break

        if found:
            s.kalista_targets[index][2] += 1
        else:
            s.kalista_targets.append([champion, target, 1])
            index = len(s.kalista_targets) - 1

        damage = champion.AD
        if target.armor >= 0:
            damage = damage * (100/(100+target.armor))
        else:
            damage = damage * (2 - 100/(100 - target.armor))

        spear_damage = stats.ACTIVE_TARGET_HEALT_THRESHOLD[champion.name][champion.stars] * target.max_health * s.kalista_targets[index][2]
        spear_damage_original = spear_damage
        if target.MR >= 0:
            spear_damage = spear_damage * (100/(100+target.MR)) * champion.SP
        else:
            spear_damage = spear_damage * (2- 100/(100 - target.MR)) * champion.SP

        targe_total_hp = target.health + target.shield_amount()
        if not target.immune and spear_damage + damage > targe_total_hp:
            champion.print(' active triggered')
            champion.spell(target, spear_damage_original)

        return {'damage': 0, 'true_damage': False, 'crit_random': None, 'dodge_random': None}


# get the change through before the first attacks come in.
# the que system only wanted to execute this after the first attacks
def tahmkench_init(champion):
    champion.damage_reduction = stats.ACTIVE_DAMAGE_REDUCTION[champion.name][champion.stars] * champion.SP
    champion.print(' {} {} --> {}'.format('damage_reduction', 0, champion.damage_reduction))


def vayne(champion, target):
    s = get_state()
    found = False
    index = -1
    if len(s.vayne_targets) > 0:
        for i, v in enumerate(s.vayne_targets):
            if v[0] == champion and v[1] == target:
                found = True
                index = i
                break

    if found:
        s.vayne_targets[index][2] += 1
    else:
        s.vayne_targets.append([champion, target, 1])
        index = len(s.vayne_targets) - 1

    if s.vayne_targets[index][2] == stats.ACTIVATE_EVERY_X_ATTACKS[champion.name]:
        s.vayne_targets[index][2] = 0
        champion.print(' active triggered')
        return {'damage': stats.ACTIVE_DMG[champion.name][champion.stars] * champion.SP,
                'true_damage': True, 'crit_random': None, 'dodge_random': None}

    else:
        return {'damage': 0, 'true_damage': True, 'crit_random': None, 'dodge_random': None}


def warwick(champion, target):
    if champion.ability_active:
        if not target:
            field.find_target(champion)
            target = champion.target
        
        damage = champion.AD

        dodge_random = random.randint(1, 100)/100
        crit_random = random.randint(1, 100)/100

        if target.armor >= 0:
            damage = damage * (100/(100+target.armor))
        else:
            damage = damage * (2 - 100/(100 - target.armor))

        if dodge_random < target.dodge:
            damage = 0

        damage *= target.receive_increased_damage

        damage -= target.damage_reduction
        if damage < 0:
            damage = 0
        if target.immune or target.autoimmune:
            damage = 0

        if crit_random < champion.crit_chance:
            damage *= champion.crit_damage

        shield = target.shield_amount()
        if damage > shield + target.health:

            # AS gain for allies who share traits with ww
            ww_traits = origin_class_stats.origin_class['warwick']
            for o in champion.own_team():
                if o != champion:

                    for t in origin_class_stats.origin_class[o.name]:
                        if t in ww_traits:

                            champion.print(o.name)
                            as_gain = stats.ABILITY_AS_SECONDARY_GAIN['warwick'][champion.stars]
                            o.add_que('change_stat', -1, None, 'AS', o.AS * as_gain)
                            o.add_que('change_stat', stats.ABILITY_LENGTH[champion.name],
                                      None, 'AS', None, {'ezreal': as_gain})
                            break

            # howling
            """ 
            target_neighbors = field.find_neighbors(target.y, target.x)
            coords = field.coordinates
            for t in target_neighbors:
                c = coords[t[0]][t[1]]
                if(c and c.team != champion.team and c.champion):
                    c.add_que('change_stat', -1, None, 'stunned', True)
                    c.clear_que_stunned_removal()
                    c.add_que('change_stat', stats.ABILITY_STUN_DURATION[champion.name][champion.stars], 
                               None, 'stunned', False)
            """

        return {'damage': 0, 'true_damage': False, 'crit_random': crit_random, 'dodge_random': dodge_random}
    else:
        return {'damage': 0, 'true_damage': False, 'crit_random': None, 'dodge_random': None}


def zed(champion, target):
    s = get_state()

    found = False
    index = -1
    if len(s.zed_counter) > 0:
        for i, v in enumerate(s.zed_counter):
            if v[0] == champion:
                found = True
                index = i
                break

    if found:
        s.zed_counter[index][1] += 1
    else:
        s.zed_counter.append([champion, 1])
        index = len(s.zed_counter) - 1

    if s.zed_counter[index][1] == stats.ACTIVATE_EVERY_X_ATTACKS[champion.name] and champion.target:
        champion.print(' active triggered')
        s.zed_counter[index][1] = 0

        stolen_ad = champion.target.AD * stats.ACTIVE_STOLEN_AD[champion.name][champion.stars]
        
        old_target_ad = champion.target.AD
        champion.target.AD -= stolen_ad
        champion.target.print(' {} {} --> {}'.format('AD', round(old_target_ad, 1), round(champion.target.AD, 1)))

        old_champion_ad = champion.AD
        champion.AD += stolen_ad
        champion.print(' {} {} --> {}'.format('AD', round(old_champion_ad, 1), round(champion.AD, 1)))

        champion.spell(champion.target, stats.ACTIVE_DMG[champion.name][champion.stars])

    return {'damage': 0, 'true_damage': False, 'crit_random': None, 'dodge_random': None}
