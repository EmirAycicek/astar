"""
A* algoritması için heuristic fonksiyonları
Hazır fonksiyon kullanmadan yazılmıştır
"""

import math

class Heuristics:
    """Heuristic fonksiyonları sınıfı"""
    
    @staticmethod
    def manhattan_distance(node1, node2):
        """
        Manhattan (Taksi) mesafesi
        Sadece yatay ve dikey hareket için ideal
        h(n) = |x1 - x2| + |y1 - y2|
        """
        return abs(node1.x - node2.x) + abs(node1.y - node2.y)
    
    @staticmethod
    def euclidean_distance(node1, node2):
        """
        Euclidean (Öklid) mesafesi
        Çapraz hareket mümkün olduğunda daha doğru
        h(n) = sqrt((x1-x2)² + (y1-y2)²)
        """
        dx = node1.x - node2.x
        dy = node1.y - node2.y
        return math.sqrt(dx * dx + dy * dy)
    
    @staticmethod
    def chebyshev_distance(node1, node2):
        """
        Chebyshev mesafesi
        8 yönlü hareket için ideal (kral hareketi)
        h(n) = max(|x1-x2|, |y1-y2|)
        """
        dx = abs(node1.x - node2.x)
        dy = abs(node1.y - node2.y)
        return max(dx, dy)
    
    @staticmethod
    def octile_distance(node1, node2):
        """
        Octile (8-yönlü) mesafesi
        Çapraz ve düz hareketin farklı maliyetli olduğu durumlarda
        Çapraz hareket √2 ≈ 1.414, düz hareket 1
        """
        dx = abs(node1.x - node2.x)
        dy = abs(node1.y - node2.y)
        
        # Çapraz ve düz hareket sayıları
        diagonal = min(dx, dy)
        straight = max(dx, dy) - diagonal
        
        # √2 ≈ 1.414 ama integer arithmetic için 14/10 kullanabiliriz
        return diagonal * 14 + straight * 10
    
    @staticmethod
    def hamming_distance(node1, node2):
        """
        Hamming mesafesi
        Koordinatların kaç tanesinin farklı olduğunu sayar
        """
        distance = 0
        if node1.x != node2.x:
            distance += 1
        if node1.y != node2.y:
            distance += 1
        return distance
    
    @staticmethod
    def weighted_euclidean(node1, node2, weight=1.0):
        """
        Ağırlıklı Euclidean mesafesi
        Weight > 1: Daha agresif arama (hızlı ama optimal olmayabilir)
        Weight < 1: Daha konservatif arama (yavaş ama daha optimal)
        """
        dx = node1.x - node2.x
        dy = node1.y - node2.y
        return weight * math.sqrt(dx * dx + dy * dy)
    
    @staticmethod
    def canberra_distance(node1, node2):
        """
        Canberra mesafesi
        Koordinat farklarının toplamlarına oranı
        """
        dx = abs(node1.x - node2.x)
        dy = abs(node1.y - node2.y)
        
        sum_x = abs(node1.x) + abs(node2.x)
        sum_y = abs(node1.y) + abs(node2.y)
        
        result = 0
        if sum_x != 0:
            result += dx / sum_x
        if sum_y != 0:
            result += dy / sum_y
        
        return result
    
    @staticmethod
    def minkowski_distance(node1, node2, p=3):
        """
        Minkowski mesafesi
        p=1: Manhattan, p=2: Euclidean
        p büyüdükçe Chebyshev'e yaklaşır
        """
        dx = abs(node1.x - node2.x)
        dy = abs(node1.y - node2.y)
        
        return (dx ** p + dy ** p) ** (1.0 / p)


class HeuristicSelector:
    """Heuristic seçici sınıfı"""
    
    def __init__(self):
        self.heuristics = {
            'manhattan': Heuristics.manhattan_distance,
            'euclidean': Heuristics.euclidean_distance,
            'chebyshev': Heuristics.chebyshev_distance,
            'octile': Heuristics.octile_distance,
            'hamming': Heuristics.hamming_distance,
            'weighted_euclidean': lambda n1, n2: Heuristics.weighted_euclidean(n1, n2, 1.2),
            'canberra': Heuristics.canberra_distance,
            'minkowski': lambda n1, n2: Heuristics.minkowski_distance(n1, n2, 3)
        }
    
    def get_heuristic(self, name):
        """İsme göre heuristic fonksiyonu döndür"""
        return self.heuristics.get(name, self.heuristics['euclidean'])
    
    def get_all_names(self):
        """Tüm heuristic isimlerini döndür"""
        return list(self.heuristics.keys())
    
    def compare_heuristics(self, node1, node2):
        """Tüm heuristic'leri karşılaştır"""
        results = {}
        for name, func in self.heuristics.items():
            try:
                results[name] = func(node1, node2)
            except:
                results[name] = float('inf')
        return results


# Heuristic'lerin özellikleri hakkında bilgi
HEURISTIC_INFO = {
    'manhattan': {
        'name': 'Manhattan Distance',
        'description': 'Sadece yatay/dikey hareket. Grid-based için ideal.',
        'admissible': True,
        'consistent': True,
        'best_for': '4-yönlü hareket'
    },
    'euclidean': {
        'name': 'Euclidean Distance', 
        'description': 'Gerçek düz çizgi mesafesi. Çapraz hareket için iyi.',
        'admissible': True,
        'consistent': True,
        'best_for': 'Serbest hareket, robotik'
    },
    'chebyshev': {
        'name': 'Chebyshev Distance',
        'description': 'Kral hareketi (satranç). 8-yönlü hareket.',
        'admissible': True,
        'consistent': True,
        'best_for': '8-yönlü eşit maliyetli hareket'
    },
    'octile': {
        'name': 'Octile Distance',
        'description': 'Çapraz ve düz hareketin farklı maliyetli olduğu durumlar.',
        'admissible': True,
        'consistent': True,
        'best_for': 'Gerçekçi 8-yönlü hareket'
    },
    'hamming': {
        'name': 'Hamming Distance',
        'description': 'Farklı koordinat sayısı. Basit durumlar için.',
        'admissible': False,
        'consistent': False,
        'best_for': 'Özel durumlar'
    }
}